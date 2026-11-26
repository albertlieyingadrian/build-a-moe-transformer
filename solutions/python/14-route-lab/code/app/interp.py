"""Looking inside a trained MoE (stages 14-16): who goes where, does it mean anything, and what breaks without it."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from app.model import GPT
from app.tokenizer import get_batch

CLASSES = ["space", "newline", "vowel", "consonant", "upper", "punct"]


def char_class(c: str) -> str:
    if c == " ":
        return "space"
    if c == "\n":
        return "newline"
    if c.isupper():
        return "upper"
    if c in "aeiou":
        return "vowel"
    if c.isalpha():
        return "consonant"
    return "punct"


@torch.no_grad()
def routing_records(model: GPT, data, gen, batches=20, batch_size=32):
    """Run the model on val windows. Returns the input token ids (N,), the next-token loss at each position (N,), and
    for every MoE layer the top-1 routed expert (N,) and all top_k experts (N, k)."""
    model.eval()
    toks, losses, top1, topk = [], [], [[] for _ in model.moe_layers()], [[] for _ in model.moe_layers()]
    for _ in range(batches):
        x, y = get_batch(data, model.cfg.block_size, batch_size, gen)
        logits, _ = model(x)
        toks.append(x.flatten())
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none"))
        for i, m in enumerate(model.moe_layers()):
            idx = m.stats["idx"]
            top1[i].append(idx[:, 0])
            topk[i].append(idx)
    return torch.cat(toks), torch.cat(losses), [torch.cat(t) for t in top1], [torch.cat(t) for t in topk]


def mutual_information(a: torch.Tensor, b: torch.Tensor) -> float:
    """I(a; b) in bits, from counts. How many bits knowing the token tells you about the expert it goes to."""
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    joint = torch.bincount(a * nb + b, minlength=na * nb).float().view(na, nb)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    nz = p > 0
    return (p[nz] * (p[nz] / (pa @ pb)[nz]).log2()).sum().item()


def entropy_bits(b: torch.Tensor) -> float:
    p = torch.bincount(b).float()
    p = p[p > 0] / p.sum()
    return -(p * p.log2()).sum().item()


def specialisation(toks, experts, gen) -> dict:
    """MI(token; expert) / H(expert): 0 means routing ignores the token, 1 means the token decides the expert.
    The control shuffles the expert labels across positions, which keeps both distributions and breaks the link."""
    h = entropy_bits(experts)
    mi = mutual_information(toks, experts)
    shuffled = experts[torch.randperm(len(experts), generator=gen)]
    return {
        "mi_bits": round(mi, 3),
        "h_expert_bits": round(h, 3),
        "normalised": round(mi / h, 3) if h > 0 else 0.0,
        "shuffled_control": round(mutual_information(toks, shuffled) / h, 3) if h > 0 else 0.0,
    }


def class_table(toks, topk_experts, chars, n_experts) -> list[list[float]]:
    """rows = character classes, columns = experts: the share of that class's top_k slots that went to each expert."""
    cls = torch.tensor([CLASSES.index(char_class(chars[t])) for t in toks.tolist()])
    table = torch.zeros(len(CLASSES), n_experts)
    for slot in range(topk_experts.shape[1]):
        table.index_put_((cls, topk_experts[:, slot]), torch.ones(len(cls)), accumulate=True)
    return (table / table.sum(1, keepdim=True).clamp(min=1)).tolist()


def top_tokens(toks, experts, chars, n_experts, k=8) -> list[str]:
    """For each expert, the characters most over-represented among its tokens (share at expert / share overall)."""
    V = len(chars)
    overall = torch.bincount(toks, minlength=V).float()
    overall = overall / overall.sum()
    out = []
    for e in range(n_experts):
        c = torch.bincount(toks[experts == e], minlength=V).float()
        if c.sum() == 0:
            out.append("")
            continue
        lift = (c / c.sum()) / overall.clamp(min=1e-9)
        lift[c < 5] = 0  # ignore characters seen only a handful of times here
        out.append("".join(chars[i] for i in lift.argsort(descending=True)[:k].tolist()).replace("\n", "⏎"))
    return out


@torch.no_grad()
def loss_on(model: GPT, data, seed, batches=20, batch_size=32):
    """Per-position next-token loss on a fixed set of val windows (the same windows for every call with this seed)."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    return torch.cat(
        [
            F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten(), reduction="none")
            for x, y in (get_batch(data, model.cfg.block_size, batch_size, gen) for _ in range(batches))
        ]
    )


def plot_heatmap(tables: list, path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(3.2 * len(tables), 2.8), squeeze=False)
    for i, (ax, t) in enumerate(zip(axes[0], tables, strict=True)):
        im = ax.imshow(t, cmap="viridis", vmin=0, vmax=max(max(r) for r in t), aspect="auto")
        ax.set_yticks(range(len(CLASSES)), CLASSES if i == 0 else [""] * len(CLASSES))
        ax.set_xticks(range(len(t[0])))
        ax.set_xlabel("expert")
        ax.set_title(f"layer {i}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title or "share of each character class's routing slots, per expert", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def collapse_flags(stats: dict, entropy_floor=0.75, max_load_ceiling=2.0) -> list[str]:
    """The checks a training dashboard would run on one layer's load stats (from train.load_stats)."""
    flags = []
    if stats["entropy"] < entropy_floor:
        flags.append(f"entropy {stats['entropy']:.2f} < {entropy_floor}")
    if stats["max_load"] > max_load_ceiling:
        flags.append(f"max load {stats['max_load']:.2f} > {max_load_ceiling}")
    if stats["dead"] > 0:
        flags.append(f"{stats['dead']} dead expert(s)")
    return flags
