"""Build a DeepSeek-style Mixture-of-Experts transformer: one command per stage.

    ./your_program.sh <command> [options]

Stage 01 tokenize  02 forward   03 train-dense  04 expert   05 router  06 dispatch  07 capacity  08 aux-loss
      09 bias-balance  10 shared-expert  11 fine-grained  12 mla  13 compare  14 route-lab  15 ablate  16 collapse
Extension: 17 mtp (multi-token prediction)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

MAX_STAGE = 1  # commands from later stages are hidden; the staged solutions lower this number

torch.set_num_threads(4)


def out_path(args, name):
    path = Path(args.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


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


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
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
