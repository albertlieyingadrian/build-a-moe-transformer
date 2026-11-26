"""Training a tiny language model, dense or MoE (stages 03, 13, 16).

The loop is the standard one (sample a batch, compute the loss, backpropagate, step) with two MoE additions:
- with balance == "bias", every router's bias is nudged after each step from the loads it just saw (stage 09);
- every eval also records how evenly each layer spread its tokens, so collapse is visible as it happens (stage 16).
"""

from __future__ import annotations

import math
import time

import torch

from app.model import GPT
from app.tokenizer import get_batch


def load_stats(counts: torch.Tensor) -> dict:
    """How evenly one layer spread its token slots over its experts.
    max_load: the busiest expert's share × E (1.0 = perfectly even, E = everything on one expert).
    entropy: entropy of the load shares / log E (1.0 = even, 0.0 = one expert).
    dead: experts that got under a tenth of their fair share."""
    E = counts.numel()
    f = counts.float() / counts.sum().clamp(min=1)
    nz = f[f > 0]
    return {
        "max_load": round((f.max() * E).item(), 4),
        "entropy": round((-(nz * nz.log()).sum() / math.log(E)).item(), 4),
        "dead": int((f < 0.1 / E).sum().item()),
        "shares": [round(v, 4) for v in f.tolist()],
    }


@torch.no_grad()
def evaluate(model: GPT, data, gen, batches=10, batch_size=32) -> dict:
    """Mean next-character loss on `batches` random windows, and each MoE layer's load over those windows."""
    model.eval()
    block = model.cfg.block_size
    losses, counts = [], [torch.zeros(m.cfg.n_routed, dtype=torch.long) for m in model.moe_layers()]
    for _ in range(batches):
        x, y = get_batch(data, block, batch_size, gen)
        logits, _ = model(x)
        losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
        for c, m in zip(counts, model.moe_layers(), strict=True):
            c += m.stats["wanted"]
    model.train()
    return {"loss": sum(losses) / len(losses), "layers": [load_stats(c) for c in counts]}


def train_lm(model: GPT, train_data, val_data, steps=1000, batch_size=32, lr=3e-3, seed=0, log=print, evals=10):
    """Train with AdamW and a cosine learning-rate decay. Returns a history: one row per eval."""
    gen = torch.Generator().manual_seed(seed)
    egen = torch.Generator().manual_seed(seed + 1)
    cfg = model.cfg
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    )
    history, t0 = [], time.time()
    model.train()
    for step in range(steps + 1):
        if step % max(1, steps // evals) == 0 or step == steps:
            ev = evaluate(model, val_data, egen, batches=10 if step < steps else 40)
            row = {
                "step": step,
                "val": round(ev["loss"], 4),
                "layers": ev["layers"],
                "time": round(time.time() - t0, 1),
            }
            history.append(row)
            if log:
                loads = " ".join(f"{la['max_load']:.2f}" for la in ev["layers"])
                log(f"step {step:5d}  val {ev['loss']:.3f}" + (f"  max load per layer {loads}" if loads else ""))
        if step == steps:
            break
        x, y = get_batch(train_data, cfg.block_size, batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if cfg.ffn == "moe" and cfg.balance == "bias":
            for m in model.moe_layers():
                m.router.update_bias(m.stats["wanted"], cfg.bias_rate)
    model.eval()
    return history
