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

MAX_STAGE = 17  # commands from later stages are hidden; the staged solutions lower this number

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


def cmd_capacity(args):
    from app.model import Config, MoE, capacity

    torch.manual_seed(args.seed)
    cfg = Config(
        n_embd=args.d,
        n_routed=args.experts,
        top_k=args.top_k,
        n_shared=0,
        expert_hidden=args.d,
        capacity_factor=args.capacity_factor,
    )
    moe = MoE(cfg)
    moe.router.skew[0] = args.skew  # a popular expert, so someone has to queue
    x = torch.randn(1, args.tokens, args.d)
    with torch.no_grad():
        out = moe(x)[0]
    wanted, kept = moe.stats["wanted"], moe.stats["kept"]
    cap = capacity(args.tokens, args.experts, args.top_k, args.capacity_factor)
    dropped = int((wanted - kept).sum())
    print(f"capacity per expert: {cap}")
    print(f"wanted: {wanted.tolist()}")
    print(f"kept:   {kept.tolist()}")
    print(f"dropped slots: {dropped} of {args.tokens * args.top_k} ({dropped / (args.tokens * args.top_k):.1%})")
    # a token is fully dropped if none of its slots made it in; find them from the routing
    idx = moe.stats["idx"]
    served = torch.zeros(args.tokens, dtype=torch.bool)
    for e in range(args.experts):
        tok = (idx == e).nonzero(as_tuple=True)[0][:cap]
        served[tok] = True
    lost = (~served).nonzero().flatten()
    print(f"tokens with every slot dropped: {len(lost)}")
    print(f"their MoE output (the residual carries them on unchanged): {out[lost].abs().sum().item():.1f}")


def cmd_aux_loss(args):
    from app.model import Router, load_balance_loss

    torch.manual_seed(args.seed)
    E, k, N = args.experts, args.top_k, args.tokens
    even_idx = torch.arange(N * k).view(N, k) % E
    print(f"aux loss, even routing: {load_balance_loss(torch.full((N, E), 1 / E), even_idx, E).item():.3f}")
    collapsed = torch.zeros(N, E)
    collapsed[:, :k] = 1 / k
    c_idx = torch.arange(k).expand(N, k)
    print(
        f"aux loss, collapsed onto {k} experts: {load_balance_loss(collapsed, c_idx, E).item():.3f} (E / k = {E / k})"
    )
    r = Router(args.d, E, k)
    r.skew[0] = 2.0  # start lopsided: expert 0 is everyone's favourite
    x = torch.randn(N, args.d)
    opt = torch.optim.SGD(r.parameters(), lr=args.lr)
    before = None
    for step in range(args.steps + 1):
        idx, _, probs = r(x)
        loss = load_balance_loss(probs, idx, E)
        share = torch.bincount(idx.flatten(), minlength=E).float() / idx.numel()
        if step == 0:
            before = (loss.item(), share[0].item())
            print(f"random router with a head start for expert 0: aux loss {loss.item():.3f}")
        if step == args.steps:
            break
        opt.zero_grad()
        loss.backward()
        if step == 0:
            print(f"router gets a gradient: {r.proj.weight.grad.abs().sum().item() > 0}")
        opt.step()
    print(f"after {args.steps} steps on the aux loss alone: {before[0]:.3f} -> {loss.item():.3f}")
    print(f"expert 0's share of slots: {before[1]:.3f} -> {share[0].item():.3f} (fair share {1 / E:.3f})")


def cmd_bias_balance(args):
    from app.model import Router
    from app.train import load_stats

    torch.manual_seed(args.seed)
    E, k = args.experts, args.top_k
    r = Router(args.d, E, k)
    torch.nn.init.normal_(r.proj.weight, std=0.3)
    r.skew[0] = 2.0  # expert 0 starts as everyone's favourite
    gen = torch.Generator().manual_seed(args.seed)
    print("step  max load  entropy  bias[0]")
    first = last = None
    for step in range(args.steps + 1):
        x = torch.randn(args.tokens, args.d, generator=gen)  # a fresh batch each step; the router never trains
        idx, w, probs = r(x)
        s = load_stats(torch.bincount(idx.flatten(), minlength=E))
        first = first or s
        last = s
        if step % max(1, args.steps // 5) == 0:
            print(f"{step:4d}  {s['max_load']:8.2f}  {s['entropy']:7.3f}  {r.bias[0].item():+.3f}")
        r.update_bias(torch.bincount(idx.flatten(), minlength=E), args.rate)
    print(f"max load: {first['max_load']:.2f} -> {last['max_load']:.2f}")
    plain = probs.gather(-1, idx)
    print(f"gate weights ignore the bias: {torch.allclose(w, plain / plain.sum(-1, keepdim=True))}")
    print(f"bias is trained by gradients: {any(p is r.bias for p in r.parameters())}")


def cmd_shared_expert(args):
    from app.model import Config, MoE, n_params

    torch.manual_seed(args.seed)
    cfg = Config(n_embd=args.d, n_routed=args.experts, top_k=args.top_k, n_shared=1, expert_hidden=args.h)
    moe = MoE(cfg)
    x = torch.randn(1, 16, args.d)
    routed, shared, router = n_params(moe.experts), n_params(moe.shared), n_params(moe.router)
    per_expert = n_params(moe.experts[0])
    print(f"parameters: routed {routed}, shared {shared}, router {router}, total {routed + shared + router}")
    print(f"active per token: {args.top_k} routed + 1 shared + router = {(args.top_k) * per_expert + shared + router}")
    with torch.no_grad():
        full = moe(x)
        moe.ablated = set(range(args.experts))
        only_shared = moe(x)
        moe.ablated, moe.ablate_shared = set(), True
        only_routed = moe(x)
    print(
        f"routed experts off: the layer is the shared expert: {torch.allclose(only_shared, moe.shared(x), atol=1e-6)}"
    )
    print(f"layer = shared part + routed part: {torch.allclose(full, only_shared + only_routed, atol=1e-6)}")


def cmd_fine_grained(args):
    from app.model import Config, MoE, fine_grained, n_params

    base = Config(n_embd=args.d, n_routed=args.experts, top_k=args.top_k, n_shared=1, expert_hidden=args.h)
    print("split  experts  top_k  shared  hidden  expert weights  router  active params  combinations")
    for m in (1, args.split):
        c = fine_grained(base, m)
        moe = MoE(c)
        experts = n_params(moe.experts) + n_params(moe.shared)
        active = c.top_k * n_params(moe.experts[0]) + n_params(moe.shared) + n_params(moe.router)
        print(
            f"{m:5d}  {c.n_routed:7d}  {c.top_k:5d}  {c.n_shared:6d}  {c.expert_hidden:6d}  {experts:14d}  "
            f"{n_params(moe.router):6d}  {active:13d}  {math.comb(c.n_routed, c.top_k)}"
        )


def cmd_mla(args):
    from app.model import MLA, Attention, Config, n_params, rope

    torch.manual_seed(args.seed)
    cfg = Config(n_embd=args.d, n_head=args.heads, kv_rank=args.kv_rank, rope_dim=args.rope_dim)
    mha, mla = Attention(cfg), MLA(cfg)
    a, b = mha.cache_floats_per_token(), mla.cache_floats_per_token()
    print(f"cache floats per token per layer: MHA {a}, MLA {b} ({a / b:.1f}x smaller)")
    print(f"parameters: MHA {n_params(mha)}, MLA {n_params(mla)}")
    x = torch.randn(1, args.T, args.d)
    with torch.no_grad():
        from app.model import causal_softmax

        s, _ = mla.scores(x)
        print(f"MLA weight on future positions: {causal_softmax(s).triu(1).abs().sum().item():.1f}")
        x2 = x.clone()
        x2[0, -1] += 10
        print(f"change at earlier positions after editing the last one: {(mla(x2) - mla(x))[0, :-1].abs().max():.1f}")
        # RoPE: rotate a query and a key placed at positions (i, j), then at (i + 5, j + 5): the score must not move
        q, k = torch.randn(1, 1, 1, args.rope_dim), torch.randn(1, 1, 1, args.rope_dim)
        qs = torch.cat([q] * args.T, 1)
        ks = torch.cat([k] * args.T, 1)
        s0 = rope(qs)[0, 7, 0] @ rope(ks)[0, 3, 0]
        s5 = rope(qs, start=5)[0, 7, 0] @ rope(ks, start=5)[0, 3, 0]
        print(f"RoPE score depends only on the distance: {torch.allclose(s0, s5, atol=1e-5)}")


# ------------------------------------------------------------------------------- stages 13-16: train and look inside
def variant_cfg(name, args, tok):
    from app.model import fine_grained

    if name == "dense":  # same active parameters as the MoE: hidden = (top_k + shared) × expert_hidden
        return model_cfg(args, tok, ffn="dense")
    if name == "dense-wide":  # same TOTAL parameters as the MoE: hidden = (experts + shared) × expert_hidden
        return model_cfg(args, tok, ffn="dense", dense_hidden=(args.experts + args.shared) * args.expert_hidden)
    if name.startswith("moe-"):
        parts = name.split("-")  # moe-<balance>[-fine]
        cfg = model_cfg(args, tok, ffn="moe", balance=parts[1])
        return fine_grained(cfg, 4) if "fine" in parts else cfg
    raise ValueError(f"unknown variant {name}")


def plot_curves(results, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, r in results.items():
        h = r["history"]
        ax.plot([row["step"] for row in h], [row["val"] for row in h], label=f"{name} ({r['val']:.3f})")
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    lo = min(r["val"] for r in results.values())
    ax.set_ylim(lo - 0.05, lo + 0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def cmd_compare(args):
    from app.model import GPT, n_params
    from app.train import train_lm

    tok, train, val = load_text(args.corpus)
    results = {}
    for name in args.variants.split(","):
        torch.manual_seed(args.seed)
        model = GPT(variant_cfg(name, args, tok))
        t = time.time()
        log = print if args.verbose else None
        hist = train_lm(
            model, train, val, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed, log=log
        )
        layers = hist[-1]["layers"]
        results[name] = {
            "params": n_params(model),
            "active": model.active_params(),
            "val": hist[-1]["val"],
            "max_load": round(sum(la["max_load"] for la in layers) / len(layers), 3) if layers else None,
            "seconds": round(time.time() - t, 1),
            "history": hist,
        }
        print(f"trained {name}: val {hist[-1]['val']:.3f} in {time.time() - t:.0f} s")
        save_ckpt(out_path(args, f"{name}_s{args.seed}.pt"), model, tok, hist)
    print()
    print("variant          total params  active params  val loss  max load")
    for name, r in results.items():
        ml = f"{r['max_load']:.2f}" if r["max_load"] is not None else "-"
        print(f"{name:15s}  {r['params']:12d}  {r['active']:13d}  {r['val']:8.3f}  {ml:>8s}")
    save_json(out_path(args, f"compare_s{args.seed}.json"), results)
    plot_curves(results, out_path(args, f"compare_s{args.seed}.png"), f"{args.steps} steps, seed {args.seed}")


def load_model(path):
    from app.model import GPT, Config
    from app.tokenizer import CharTokenizer

    ck = torch.load(path, weights_only=False)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.eval()
    tok = CharTokenizer("")
    tok.chars, tok.stoi = ck["chars"], {c: i for i, c in enumerate(ck["chars"])}
    return model, tok


def cmd_route_lab(args):
    from app.interp import CLASSES, class_table, plot_heatmap, routing_records, specialisation, top_tokens
    from app.model import GPT, Config
    from app.train import load_stats

    model, tok = load_model(args.ckpt)
    _, _, val = load_text(args.corpus)
    gen = torch.Generator().manual_seed(args.seed)
    toks, _, top1, topk = routing_records(model, val, gen, batches=args.batches)
    torch.manual_seed(args.seed)
    fresh = GPT(Config(**model.cfg.to_dict()))
    _, _, fresh_top1, _ = routing_records(fresh, val, torch.Generator().manual_seed(args.seed), batches=args.batches)
    E = model.cfg.n_routed
    report, tables = [], []
    for i in range(len(top1)):
        spec = specialisation(toks, top1[i], gen)
        spec["untrained_model"] = specialisation(toks, fresh_top1[i], gen)["normalised"]
        loads = load_stats(torch.bincount(topk[i].flatten(), minlength=E))
        table = class_table(toks, topk[i], tok.chars, E)
        tables.append(table)
        tops = top_tokens(toks, top1[i], tok.chars, E)
        print(f"layer {i}: load shares {[round(s, 2) for s in loads['shares']]}  max load {loads['max_load']:.2f}")
        print(
            f"  token -> expert: {spec['normalised']:.3f} of the expert's entropy "
            f"(shuffled control {spec['shuffled_control']:.3f}, untrained model {spec['untrained_model']:.3f})"
        )
        for e in range(E):
            best = max(range(len(CLASSES)), key=lambda c: table[c][e])
            print(f"  expert {e}: most over-represented {tops[e]!r:14s} biggest class share: {CLASSES[best]}")
        report.append({"layer": i, "specialisation": spec, "load": loads, "class_table": table, "top_tokens": tops})
    save_json(out_path(args, "route_lab.json"), report)
    plot_heatmap(tables, out_path(args, "routing_heatmap.png"))


def cmd_ablate(args):
    from app.interp import loss_on, routing_records

    model, _ = load_model(args.ckpt)
    _, _, val = load_text(args.corpus)
    base = loss_on(model, val, args.seed, batches=args.batches)
    _, _, _, topk = routing_records(model, val, torch.Generator().manual_seed(args.seed), batches=args.batches)
    print(f"val loss, nothing ablated: {base.mean():.3f}")
    rows, targeted = [], []
    for li, m in enumerate(model.moe_layers()):
        for e in range(model.cfg.n_routed):
            m.ablated = {e}
            delta = loss_on(model, val, args.seed, batches=args.batches) - base
            m.ablated = set()
            mine = (topk[li] == e).any(-1)
            own, other = delta[mine].mean().item(), delta[~mine].mean().item()
            rows.append(
                {
                    "layer": li,
                    "expert": e,
                    "share": mine.float().mean().item(),
                    "delta": delta.mean().item(),
                    "own_tokens": own,
                    "other_tokens": other,
                }
            )
            targeted.append(own > other)
        m.ablate_shared = True
        d = (loss_on(model, val, args.seed, batches=args.batches) - base).mean().item()
        m.ablate_shared = False
        rows.append({"layer": li, "expert": "shared", "share": 1.0, "delta": d, "own_tokens": d, "other_tokens": 0.0})
    print("layer  expert  share of tokens  loss change  on its own tokens  on the rest")
    for r in rows:
        print(
            f"{r['layer']:5d}  {str(r['expert']):>6s}  {r['share']:15.2f}  {r['delta']:+11.3f}  "
            f"{r['own_tokens']:+17.3f}  {r['other_tokens']:+11.3f}"
        )
    routed = [r["delta"] for r in rows if r["expert"] != "shared"]
    shared = [r["delta"] for r in rows if r["expert"] == "shared"]
    print(f"mean loss change, one routed expert off: {sum(routed) / len(routed):+.3f}")
    print(f"mean loss change, the shared expert off: {sum(shared) / len(shared):+.3f}")
    print(f"ablations that hurt an expert's own tokens more than the rest: {sum(targeted)} of {len(targeted)}")
    save_json(out_path(args, "ablate.json"), rows)


def cmd_collapse(args):
    from app.interp import collapse_flags
    from app.model import GPT
    from app.train import load_stats, train_lm

    print(f"detector on an even layer: {collapse_flags(load_stats(torch.tensor([25, 25, 25, 25]))) or 'ok'}")
    print(f"detector on a collapsed layer: {collapse_flags(load_stats(torch.tensor([97, 1, 1, 1])))}")
    tok, train, val = load_text(args.corpus)
    runs = {
        "none": (0.0, "none"),
        "none-skew": (args.skew, "none"),
        "aux-skew": (args.skew, "aux"),
        "bias-skew": (args.skew, "bias"),
    }
    results = {}
    for name, (skew, balance) in runs.items():
        torch.manual_seed(args.seed)
        model = GPT(model_cfg(args, tok, ffn="moe", balance=balance, router_skew=skew))
        hist = train_lm(
            model, train, val, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed, log=None
        )
        last = hist[-1]["layers"]
        results[name] = {"val": hist[-1]["val"], "history": hist}
        worst = max(last, key=lambda la: la["max_load"])
        flags = sorted({f for la in last for f in collapse_flags(la)})
        print(
            f"{name:10s} val {hist[-1]['val']:.3f}  worst layer: max load {worst['max_load']:.2f}, entropy "
            f"{worst['entropy']:.2f}, dead {worst['dead']}  flags: {flags or 'none'}"
        )
        results[name]["worst_max_load"] = worst["max_load"]
        results[name]["first_flagged_step"] = next(
            (row["step"] for row in hist if any(collapse_flags(la) for la in row["layers"])), None
        )
    for name, r in results.items():
        print(
            f"{name:10s} max load per eval (worst layer): "
            + " ".join(f"{max(la['max_load'] for la in row['layers']):.1f}" for row in r["history"])
        )
    save_json(out_path(args, f"collapse_s{args.seed}.json"), results)
    plot_collapse(results, out_path(args, f"collapse_s{args.seed}.png"), args)


def plot_collapse(results, path, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, r in results.items():
        h = r["history"]
        ax.plot(
            [row["step"] for row in h],
            [max(la["max_load"] for la in row["layers"]) for row in h],
            "-o",
            ms=3,
            label=name,
        )
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("max load, worst layer (1 = even)")
    ax.set_title(f"routing collapse and recovery ({args.experts} experts, top-{args.top_k})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------------------------------------------- extension 17: MTP
def cmd_mtp(args):
    import torch.nn.functional as F

    from app.model import GPT, n_params
    from app.tokenizer import get_batch
    from app.train import train_lm

    tok, train, val = load_text(args.corpus)
    out = {}
    for mtp in (0, 1):
        torch.manual_seed(args.seed)
        model = GPT(model_cfg(args, tok, ffn="moe", mtp=mtp))
        hist = train_lm(
            model, train, val, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed, log=None
        )
        out[mtp] = {"val": hist[-1]["val"], "active": model.active_params(), "params": n_params(model)}
        print(f"mtp={mtp}: main val loss {hist[-1]['val']:.3f}, active parameters {model.active_params()}")
        if mtp:
            gen = torch.Generator().manual_seed(args.seed + 7)
            hits = agree = n = 0
            losses = []
            with torch.no_grad():
                for _ in range(20):
                    x, y = get_batch(val, model.cfg.block_size, 32, gen)
                    h = model.hidden(x)
                    main = model.head(model.norm(h)).argmax(-1)  # main model's guess for t+1 at every t
                    draft = model.head(model.norm(model.mtp(h[:, :-1], model.tok_emb(y[:, :-1]))))
                    losses.append(F.cross_entropy(draft.flatten(0, 1), y[:, 1:].flatten()).item())
                    d = draft.argmax(-1)  # MTP's guess for t+2 at every t
                    hits += (d == y[:, 1:]).sum().item()
                    agree += (d == main[:, 1:]).sum().item()  # the main model's own guess for that same token
                    n += d.numel()
            out["mtp_val"] = sum(losses) / len(losses)
            out["mtp_acc"], out["mtp_agree"] = hits / n, agree / n
            print(f"MTP module parameters (training only): {n_params(model.mtp)}")
            print(f"MTP val loss on token t+2: {out['mtp_val']:.3f} (guessing: {math.log(tok.vocab_size):.3f})")
            print(f"MTP top-1 accuracy on token t+2: {out['mtp_acc']:.3f}")
            print(f"MTP agrees with the main model's next-step guess: {out['mtp_agree']:.3f} (a draft acceptance rate)")
    print(f"active parameters unchanged by MTP: {out[0]['active'] == out[1]['active']}")
    print(f"main val loss change with MTP: {out[1]['val'] - out[0]['val']:+.3f}")
    save_json(out_path(args, f"mtp_s{args.seed}.json"), {str(k): v for k, v in out.items()})


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
    "forward": (2, cmd_forward),
    "train-dense": (3, cmd_train_dense),
    "expert": (4, cmd_expert),
    "router": (5, cmd_router),
    "dispatch": (6, cmd_dispatch),
    "capacity": (7, cmd_capacity),
    "aux-loss": (8, cmd_aux_loss),
    "bias-balance": (9, cmd_bias_balance),
    "shared-expert": (10, cmd_shared_expert),
    "fine-grained": (11, cmd_fine_grained),
    "mla": (12, cmd_mla),
    "compare": (13, cmd_compare),
    "route-lab": (14, cmd_route_lab),
    "ablate": (15, cmd_ablate),
    "collapse": (16, cmd_collapse),
    "mtp": (17, cmd_mtp),
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
