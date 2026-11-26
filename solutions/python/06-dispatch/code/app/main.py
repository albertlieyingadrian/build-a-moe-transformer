"""Build a DeepSeek-style Mixture-of-Experts transformer: one command per stage.

    ./your_program.sh <command> [options]

Stage 01 tokenize  02 forward   03 train-dense  04 expert   05 router  06 dispatch  07 capacity  08 aux-loss
      09 bias-balance  10 shared-expert  11 fine-grained  12 mla  13 compare  14 route-lab  15 ablate  16 collapse
Extension: 17 mtp (multi-token prediction)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

MAX_STAGE = 6  # commands from later stages are hidden; the staged solutions lower this number

torch.set_num_threads(4)


def out_path(args, name):
    path = Path(args.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def load_text(path):
    from app.tokenizer import CharTokenizer, split

    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    return tok, *split(tok.encode(text))


# -------------------------------------------------------------------------------- stages 01-03: a dense baseline
def cmd_tokenize(args):
    from app.tokenizer import CharTokenizer, get_batch, split

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    tok = CharTokenizer(corpus)
    ids = tok.encode(args.text)
    print(f"corpus characters: {len(corpus)}")
    print(f"vocab size: {tok.vocab_size}")
    print(f"ids: {ids}")
    print(f"decoded: {tok.decode(ids)}")
    print(f"round trip ok: {tok.decode(ids) == args.text}")
    train, val = split(tok.encode(corpus))
    print(f"train / val characters: {len(train)} / {len(val)}")
    x, y = get_batch(train, 8, 4, torch.Generator().manual_seed(args.seed))
    print(f"batch shape: {tuple(x.shape)}")
    print(f"targets are the inputs shifted by one: {torch.equal(x[:, 1:], y[:, :-1])}")


def cmd_forward(args):
    from app.model import GPT, Config, n_params

    tok, _, _ = load_text(args.corpus)
    torch.manual_seed(args.seed)
    model = GPT(Config(vocab_size=tok.vocab_size, n_layer=2))
    ids = torch.tensor([tok.encode(args.text)])
    logits, loss = model(ids[:, :-1], ids[:, 1:])
    print(f"ids shape: {tuple(ids.shape)}")
    print(f"logits shape: {tuple(model(ids)[0].shape)}")
    print(f"parameters: {n_params(model)}")
    print(f"loss at initialisation: {loss.item():.3f}")
    print(f"ln(vocab size): {math.log(tok.vocab_size):.3f}")
    edited = ids.clone()
    edited[0, -1] = (edited[0, -1] + 1) % tok.vocab_size
    change = (model(edited)[0][0, :-1] - model(ids)[0][0, :-1]).abs().max().item()
    print(f"change at earlier positions after editing the last token: {change:.1f}")


def model_cfg(args, tok, **over):
    from app.model import Config

    d = args.n_embd
    cfg = Config(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=d,
        attn=args.attn,
        kv_rank=args.kv_rank or d // 4,
        rope_dim=args.rope_dim or d // args.heads // 2,
        n_routed=args.experts,
        top_k=args.top_k,
        n_shared=args.shared,
        expert_hidden=args.expert_hidden,
        bias_rate=args.bias_rate,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def save_ckpt(path, model, tok, history=None):
    torch.save({"cfg": model.cfg.to_dict(), "state": model.state_dict(), "chars": tok.chars, "history": history}, path)
    print(f"wrote {path}")


def cmd_train_dense(args):
    from app.model import GPT, n_params
    from app.tokenizer import CharTokenizer
    from app.train import train_lm

    tok, train, val = load_text(args.corpus)
    torch.manual_seed(args.seed)
    model = GPT(model_cfg(args, tok, ffn="dense"))
    print(f"parameters: {n_params(model)}")
    t = time.time()
    hist = train_lm(model, train, val, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(f"final val loss: {hist[-1]['val']:.3f} (guessing: ln V = {math.log(tok.vocab_size):.3f})")
    print(f"training time: {time.time() - t:.0f} s")
    prompt = torch.tensor([CharTokenizer.encode(tok, args.prompt)])
    print("sample (greedy):")
    print(tok.decode(model.generate(prompt, args.sample_len)[0].tolist()))
    save_ckpt(out_path(args, "dense.pt"), model, tok, hist)


# ------------------------------------------------------------------------------- stages 04-11: the MoE layer
def cmd_expert(args):
    import torch.nn.functional as F

    from app.model import Expert, n_params

    torch.manual_seed(args.seed)
    d, h = args.d, args.h
    ex = Expert(d, h)
    x = torch.randn(5, d)
    print(f"expert parameters: {n_params(ex)} (3 × d × h = {3 * d * h})")
    print(f"a GELU MLP (two matrices) with the same parameters has hidden width: {3 * h // 2}")
    print(f"output shape: {tuple(ex(x).shape)}")
    s = F.silu(torch.tensor([-3.0, 0.0, 3.0])).tolist()
    print(f"silu(-3), silu(0), silu(3): {s[0]:.3f} {s[1]:.3f} {s[2]:.3f}")
    with torch.no_grad():
        ex.gate.weight.zero_()
    print(f"output with the gate closed (gate weights 0): {ex(x).abs().max().item():.1f}")


def cmd_router(args):
    from app.model import Router

    torch.manual_seed(args.seed)
    r = Router(args.d, args.experts, args.top_k)
    torch.nn.init.normal_(r.proj.weight, std=0.15)
    x = torch.randn(args.tokens, args.d)
    idx, w, probs = r(x)
    for t in range(args.tokens):
        print(f"token {t}: experts {idx[t].tolist()} weights {[round(v, 2) for v in w[t].tolist()]}")
    print(f"probabilities sum to 1: {torch.allclose(probs.sum(-1), torch.ones(args.tokens))}")
    print(f"top-k weights sum to 1: {torch.allclose(w.sum(-1), torch.ones(args.tokens))}")
    print(f"experts per token distinct: {all(len(set(row)) == args.top_k for row in idx.tolist())}")
    best = probs.topk(args.top_k, -1).indices.sort(-1).values
    print(f"chosen experts are the most probable: {torch.equal(best, idx.sort(-1).values)}")
    print(f"slots per expert: {torch.bincount(idx.flatten(), minlength=args.experts).tolist()}")


def cmd_dispatch(args):
    from app.model import Config, MoE

    torch.manual_seed(args.seed)
    cfg = Config(n_embd=args.d, n_routed=args.experts, top_k=args.top_k, n_shared=0, expert_hidden=args.d)
    moe = MoE(cfg)
    torch.nn.init.normal_(moe.router.proj.weight, std=0.5)
    x = torch.randn(1, args.tokens, args.d)
    with torch.no_grad():
        fast = moe(x)[0]
        idx, w, _ = moe.router(x[0])
        slow = torch.stack(  # the obvious version: one token at a time, one chosen expert at a time
            [sum(w[t, j] * moe.experts[idx[t, j]](x[0, t]) for j in range(args.top_k)) for t in range(args.tokens)]
        )
    counts = moe.stats["wanted"]
    print(f"tokens per expert: {counts.tolist()}")
    print(f"total slots: {int(counts.sum())} (tokens × top_k = {args.tokens * args.top_k})")
    print(f"max difference from the token-by-token reference: {(fast - slow).abs().max().item():.1e}")
    one = MoE(Config(n_embd=args.d, n_routed=1, top_k=1, n_shared=0, expert_hidden=args.d))
    with torch.no_grad():
        same = torch.allclose(one(x), one.experts[0](x), atol=1e-6)
    print(f"one expert, top-1: the layer is just that expert: {same}")


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
    "forward": (2, cmd_forward),
    "train-dense": (3, cmd_train_dense),
    "expert": (4, cmd_expert),
    "router": (5, cmd_router),
    "dispatch": (6, cmd_dispatch),
}

TRAINING = ("train-dense", "compare", "collapse", "mtp")


def parser():
    p = argparse.ArgumentParser(
        prog="your_program.sh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, (stage, _) in COMMANDS.items():
        if stage > MAX_STAGE:
            continue
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out-dir", default="runs")
        if name in ("tokenize", "forward"):
            s.add_argument("corpus")
            s.add_argument("text")
        if name in TRAINING or name in ("route-lab", "ablate"):
            s.add_argument("corpus")
        if name in TRAINING:
            s.add_argument("--steps", type=int, default=3000)
            s.add_argument("--layers", type=int, default=4)
            s.add_argument("--n-embd", type=int, default=128)
            s.add_argument("--heads", type=int, default=4)
            s.add_argument("--block-size", type=int, default=128)
            s.add_argument("--batch-size", type=int, default=32)
            s.add_argument("--lr", type=float, default=3e-3)
            s.add_argument("--attn", default="mha" if name == "train-dense" else "mla")
            s.add_argument("--kv-rank", type=int, default=0, help="MLA latent width (0: n_embd / 4)")
            s.add_argument("--rope-dim", type=int, default=0, help="MLA RoPE width per head (0: head width / 2)")
            s.add_argument("--experts", type=int, default=8)
            s.add_argument("--top-k", type=int, default=2)
            s.add_argument("--shared", type=int, default=1)
            s.add_argument("--expert-hidden", type=int, default=128)
            s.add_argument("--bias-rate", type=float, default=0.01)
        if name == "train-dense":
            s.add_argument("--prompt", default="\n")
            s.add_argument("--sample-len", type=int, default=200)
        if name == "compare":
            s.add_argument("--variants", default="dense,moe-bias")
            s.add_argument("--verbose", action="store_true")
        if name == "collapse":
            s.add_argument("--skew", type=float, default=3.0)
        if name in (
            "expert",
            "router",
            "dispatch",
            "capacity",
            "aux-loss",
            "bias-balance",
            "shared-expert",
            "fine-grained",
            "mla",
        ):
            s.add_argument("--d", type=int, default=64 if name != "fine-grained" else 64)
        if name in ("expert", "shared-expert", "fine-grained"):
            s.add_argument("--h", type=int, default=128 if name == "expert" else 64)
        if name in ("router", "dispatch", "capacity", "aux-loss", "bias-balance", "shared-expert", "fine-grained"):
            s.add_argument("--experts", type=int, default=8 if name != "dispatch" else 4)
            s.add_argument("--top-k", type=int, default=2 if name != "capacity" else 1)
        if name in ("router", "dispatch", "capacity", "aux-loss", "bias-balance"):
            default_tokens = {"router": 6, "dispatch": 32, "capacity": 64, "aux-loss": 512, "bias-balance": 512}
            s.add_argument("--tokens", type=int, default=default_tokens[name])
        if name == "capacity":
            s.add_argument("--capacity-factor", type=float, default=1.0)
            s.add_argument("--skew", type=float, default=1.0)
        if name == "aux-loss":
            s.add_argument("--steps", type=int, default=100)
            s.add_argument("--lr", type=float, default=1.0)
        if name == "bias-balance":
            s.add_argument("--steps", type=int, default=200)
            s.add_argument("--rate", type=float, default=0.01)
        if name == "fine-grained":
            s.add_argument("--split", type=int, default=4)
        if name == "mla":
            s.add_argument("--heads", type=int, default=4)
            s.add_argument("--kv-rank", type=int, default=16)
            s.add_argument("--rope-dim", type=int, default=8)
            s.add_argument("--T", type=int, default=12)
        if name in ("route-lab", "ablate"):
            s.add_argument("--ckpt", default="runs/moe-bias_s0.pt")
            s.add_argument("--batches", type=int, default=20)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    sys.exit(main())
