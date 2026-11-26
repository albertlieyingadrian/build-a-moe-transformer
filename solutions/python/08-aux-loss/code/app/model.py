"""A tiny DeepSeek-style Mixture-of-Experts transformer (stages 02-12, 17).

The skeleton is a nanoGPT-shaped decoder (https://github.com/karpathy/nanoGPT), written fresh for this course. Two
pieces can be swapped, which is the whole point of the course:

    token ids ──► embed ──► x  (the residual stream: one vector of n_embd numbers per position)
                             │
                             ├──► Block:  x = x + Attention(RMSNorm(x))    "look at other positions"
                             │            x = x + FFN(RMSNorm(x))          "think about what you saw"
                             │                     │
                             │        dense:  one SwiGLU MLP for every token                     (stages 02-04)
                             │        MoE:    a router picks top_k of n_routed SwiGLU experts    (stages 05-11)
                             │                per token, plus n_shared experts every token uses
                             │
                             │        attention:  "mha" (plain multi-head) or "mla" (Multi-head Latent
                             │                     Attention with decoupled RoPE, stage 12)
                             ▼
                          RMSNorm ──► head ──► logits (a score for every possible next character)

A dense FFN is one big kitchen where every order is cooked by the same chef. An MoE FFN is a food court: a host
(the router) reads each order and sends it to the top_k stalls it suits best, so the food court can have many more
cooks in total while each order still only costs top_k cooks' time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 64  # longest sequence the model reads
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64  # width of the residual stream
    attn: str = "mha"  # "mha" (learned positions + multi-head attention) or "mla" (latent attention + RoPE)
    kv_rank: int = 16  # MLA: width of the compressed key/value latent that gets cached
    rope_dim: int = 8  # MLA: per-head width of the decoupled RoPE part of queries and keys
    ffn: str = "dense"  # "dense" or "moe"
    dense_hidden: int = 0  # dense SwiGLU hidden width (0: (top_k + n_shared) * expert_hidden, the matched width)
    n_routed: int = 8  # routed experts per MoE layer
    top_k: int = 2  # routed experts each token uses
    n_shared: int = 1  # shared experts: every token uses them, no routing
    expert_hidden: int = 64  # hidden width of one expert
    capacity_factor: float = 0.0  # 0: no limit. Otherwise each expert takes at most cf * N * top_k / n_routed slots
    balance: str = "bias"  # "none", "aux" (auxiliary loss) or "bias" (DeepSeek-V3 auxiliary-loss-free)
    aux_coef: float = 0.01  # weight of the load-balancing loss when balance == "aux"
    bias_rate: float = 0.01  # how far each bias moves per step when balance == "bias"
    router_skew: float = 0.0  # a fixed head start for expert 0's router score (stage 16's collapse experiments)
    mtp: int = 0  # 1: add a multi-token-prediction module that also predicts the token after next (stage 17)
    mtp_coef: float = 0.3  # weight of the MTP loss

    def to_dict(self):
        return asdict(self)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ------------------------------------------------------------------------------------------ stage 02: building blocks
class RMSNorm(nn.Module):
    """Rescale each vector to unit root-mean-square, then by a learned gain. LayerNorm without the mean subtraction
    and without a bias: cheaper, and what DeepSeek, Llama and most recent models use."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Rotary position embedding. x is (B, T, heads, D). Each (even, odd) pair of features is rotated by an angle
    that grows with the position, at a different speed per pair. A query at position i and a key at position j then
    meet at an angle that depends only on i - j: the dot product sees distance, not absolute place."""
    T, D = x.shape[1], x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = (torch.arange(T, device=x.device, dtype=torch.float32) + start)[:, None] * freqs  # (T, D / 2)
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    """scores is (B, heads, T, T). Hide the future (position t may not read t+1, t+2, ...), then softmax each row."""
    T = scores.shape[-1]
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device), diagonal=1)
    return F.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


class Attention(nn.Module):
    """Plain causal multi-head attention. Each position writes a query, a key and a value per head; it reads the
    values of earlier positions whose keys match its query. To generate fast, a model caches every past key and
    value: 2 * n_embd numbers per token per layer."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def cache_floats_per_token(self) -> int:
        return 2 * self.qkv.in_features  # one key and one value of n_embd numbers

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = causal_softmax(q @ k.transpose(-2, -1) / math.sqrt(k.size(-1)))
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, C))


# ---------------------------------------------------------------------------- stage 12: Multi-head Latent Attention
class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2). Instead of caching every head's key and value, cache one small
    latent vector c_kv per token and rebuild the keys and values from it. Think of storing a zip file instead of the
    folder: kv_rank numbers per token instead of 2 * n_embd.

    RoPE cannot live inside the latent (rotating by position would stop the up-projections being folded into the
    query at inference time), so position travels in a separate, "decoupled" slice: each head's query gets rope_dim
    extra numbers, and all heads share one rope_dim-wide key, which is cached too.

        x ──► W_dkv ──► c_kv (kv_rank) ──► RMSNorm ──┬─► W_uk ──► k_content (heads × head_dim)
                         [cached]                     └─► W_uv ──► v         (heads × head_dim)
        x ──► W_kr ──► RoPE ──► k_rope (rope_dim, one for all heads)  [cached]
        x ──► W_q  ──► q_content (heads × head_dim) and q_rope (heads × rope_dim) ──► RoPE on q_rope
        score = q_content · k_content + q_rope · k_rope
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.head_dim, self.rope_dim = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.rope_dim
        self.kv_rank = cfg.kv_rank
        self.w_dkv = nn.Linear(cfg.n_embd, cfg.kv_rank, bias=False)  # down-projection: the part that is cached
        self.kv_norm = RMSNorm(cfg.kv_rank)
        self.w_uk = nn.Linear(cfg.kv_rank, cfg.n_embd, bias=False)  # latent -> content keys for every head
        self.w_uv = nn.Linear(cfg.kv_rank, cfg.n_embd, bias=False)  # latent -> values for every head
        self.w_kr = nn.Linear(cfg.n_embd, cfg.rope_dim, bias=False)  # the shared positional key
        self.w_q = nn.Linear(cfg.n_embd, cfg.n_head * (self.head_dim + cfg.rope_dim), bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def cache_floats_per_token(self) -> int:
        return self.kv_rank + self.rope_dim

    def scores(self, x):
        """Attention scores (B, heads, T, T) before the mask, and the values (B, heads, T, head_dim)."""
        B, T, _ = x.shape
        H, D, R = self.n_head, self.head_dim, self.rope_dim
        c_kv = self.kv_norm(self.w_dkv(x))
        k_c = self.w_uk(c_kv).view(B, T, H, D)
        v = self.w_uv(c_kv).view(B, T, H, D)
        k_r = rope(self.w_kr(x).view(B, T, 1, R)).expand(B, T, H, R)
        q = self.w_q(x).view(B, T, H, D + R)
        q = torch.cat([q[..., :D], rope(q[..., D:])], dim=-1)
        k = torch.cat([k_c, k_r], dim=-1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        return q @ k.transpose(-2, -1) / math.sqrt(D + R), v

    def forward(self, x):
        B, T, C = x.shape
        s, v = self.scores(x)
        return self.proj((causal_softmax(s) @ v).transpose(1, 2).reshape(B, T, C))


# ------------------------------------------------------------------------------------- stage 04: the SwiGLU expert
class Expert(nn.Module):
    """A SwiGLU feed-forward network: down( silu(gate(x)) * up(x) ). The gate decides, number by number, how much of
    the up-projection to let through, like a dimmer switch on every wire. Three matrices of d × h: 3·d·h weights."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ------------------------------------------------------------------------------------------- stage 05: the router
class Router(nn.Module):
    """Score every expert for every token, pick the top_k, and weight them.

    probs = softmax(W x): one number per expert, summing to 1.
    choose = top_k of (probs + bias): the bias (stage 09) only steers the choice; it never enters the weights.
    weights = the chosen probs, rescaled to sum to 1 (when top_k > 1; with one expert the raw prob is kept, as in
    the Switch Transformer, so the router still gets a gradient).
    """

    def __init__(self, d: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.proj = nn.Linear(d, n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_experts))  # a buffer, not a Parameter: no gradient ever
        self.register_buffer("skew", torch.zeros(n_experts))  # a fixed head start, for collapse experiments

    def forward(self, x):
        probs = F.softmax(self.proj(x) + self.skew, dim=-1)  # (N, E)
        idx = (probs + self.bias).topk(self.top_k, dim=-1).indices  # (N, k): which experts
        w = probs.gather(-1, idx)  # (N, k): how much of each
        if self.top_k > 1:
            w = w / w.sum(-1, keepdim=True)
        return idx, w, probs

    @torch.no_grad()
    def update_bias(self, counts: torch.Tensor, rate: float):
        """Auxiliary-loss-free balancing (DeepSeek-V3): after each step, nudge the bias of every overloaded expert
        down and every underloaded one up by a fixed amount. A thermostat, not a penalty."""
        self.bias += rate * torch.sign(counts.float().mean() - counts.float())


# ----------------------------------------------------------------------------------- stages 06-11: the MoE layer
def capacity(n_tokens: int, n_experts: int, top_k: int, factor: float) -> int:
    """How many token slots each expert accepts: factor × its fair share (N · top_k / E), rounded up."""
    return math.ceil(factor * n_tokens * top_k / n_experts) if factor > 0 else n_tokens


def load_balance_loss(probs: torch.Tensor, idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """The Switch Transformer / DeepSeek auxiliary loss: E · Σ_i f_i · P_i.
    f_i = share of the N·k token slots sent to expert i (counted, so no gradient);
    P_i = mean router probability of expert i (differentiable).
    It is 1.0 when routing is perfectly even and grows towards E / k as routing collapses onto k experts."""
    f = torch.bincount(idx.flatten(), minlength=n_experts).float() / idx.numel()
    return n_experts * (f * probs.mean(0)).sum()


class MoE(nn.Module):
    """n_shared experts every token uses, plus n_routed experts of which the router picks top_k per token."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.n_embd, cfg.expert_hidden
        self.router = Router(d, cfg.n_routed, cfg.top_k)
        self.experts = nn.ModuleList(Expert(d, h) for _ in range(cfg.n_routed))
        # n_shared experts that see every token add up to one wider expert, so we store them as one
        self.shared = Expert(d, cfg.n_shared * h) if cfg.n_shared > 0 else None
        self.ablated: set[int] = set()  # routed experts switched off (stage 15)
        self.ablate_shared = False
        self.stats: dict = {}
        self.aux_loss = torch.zeros(())

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        N, E = flat.shape[0], self.cfg.n_routed
        idx, w, probs = self.router(flat)
        cap = capacity(N, E, self.cfg.top_k, self.cfg.capacity_factor)
        out = torch.zeros_like(flat)
        wanted, kept = torch.zeros(E, dtype=torch.long), torch.zeros(E, dtype=torch.long)
        for e in range(E):
            # dispatch: which (token, slot) pairs chose expert e, in token order
            tok, slot = (idx == e).nonzero(as_tuple=True)
            wanted[e] = len(tok)
            tok, slot = tok[:cap], slot[:cap]  # capacity: first come, first served; the rest are dropped
            kept[e] = len(tok)
            if len(tok) == 0 or e in self.ablated:
                continue
            # combine: add the expert's answer back at each token's row, scaled by its gate weight
            out.index_add_(0, tok, w[tok, slot, None] * self.experts[e](flat[tok]))
        if self.shared is not None and not self.ablate_shared:
            out = out + self.shared(flat)
        self.aux_loss = load_balance_loss(probs, idx, E)
        self.stats = {"idx": idx.detach(), "wanted": wanted, "kept": kept, "probs": probs.detach()}
        return out.view(B, T, d)


def dense_width(cfg: Config) -> int:
    return cfg.dense_hidden or (cfg.top_k + cfg.n_shared) * cfg.expert_hidden


def fine_grained(cfg: Config, m: int) -> Config:
    """DeepSeekMoE's fine-grained split: cut every expert into m narrower ones and route to m times as many.
    Same total weights, same active weights, far more ways to combine experts."""
    assert cfg.expert_hidden % m == 0
    c = Config(**cfg.to_dict())
    c.n_routed, c.top_k, c.n_shared = cfg.n_routed * m, cfg.top_k * m, cfg.n_shared * m
    c.expert_hidden = cfg.expert_hidden // m
    return c


# ------------------------------------------------------------------------------------------------------ the model
class Block(nn.Module):
    def __init__(self, cfg: Config, ffn: str | None = None):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = MLA(cfg) if cfg.attn == "mla" else Attention(cfg)
        ffn = ffn or cfg.ffn
        self.ffn = MoE(cfg) if ffn == "moe" else Expert(cfg.n_embd, dense_width(cfg))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MTP(nn.Module):
    """Multi-token prediction (DeepSeek-V3), depth 1. At position t the main model predicts token t+1. This module
    takes the main model's final state at t and the embedding of token t+1, mixes them, runs one more block, and
    predicts token t+2 with the shared output head. Like a chess player asked to also name the move after next:
    more training signal per position. At inference it can be dropped, or used to draft tokens."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm_h, self.norm_e = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.mix = nn.Linear(2 * cfg.n_embd, cfg.n_embd, bias=False)
        self.block = Block(cfg, ffn="dense")

    def forward(self, h, next_emb):
        return self.block(self.mix(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1)))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # MLA carries position inside attention (RoPE); plain attention needs a learned position table
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.attn == "mha" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.mtp = MTP(cfg) if cfg.mtp else None
        for m in self.modules():
            if isinstance(m, nn.Linear | nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.moe_layers():
            m.router.skew[0] = cfg.router_skew

    def moe_layers(self) -> list[MoE]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoE)]

    def embed(self, idx):
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        return x

    def hidden(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, idx, targets=None, reduction="mean"):
        h = self.hidden(idx)
        logits = self.head(self.norm(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction=reduction)
        if reduction != "mean":
            return logits, loss
        self.lm_loss = loss.detach()
        if self.cfg.ffn == "moe" and self.cfg.balance == "aux":
            loss = loss + self.cfg.aux_coef * sum(m.aux_loss for m in self.moe_layers())
        if self.mtp is not None and idx.shape[1] > 1:
            # position t: main state at t + embedding of token t+1 (= targets[t]) -> predict token t+2 (= targets[t+1])
            h2 = self.mtp(h[:, :-1], self.tok_emb(targets[:, :-1]))
            mtp_loss = F.cross_entropy(self.head(self.norm(h2)).flatten(0, 1), targets[:, 1:].flatten())
            self.mtp_loss = mtp_loss.detach()
            loss = loss + self.cfg.mtp_coef * mtp_loss
        return logits, loss

    def active_params(self) -> int:
        """Weights one token uses to predict the next: everything except the routed experts it was not sent to
        (and the MTP module, which only trains)."""
        skipped = sum((self.cfg.n_routed - self.cfg.top_k) * n_params(m.experts[0]) for m in self.moe_layers())
        mtp = n_params(self.mtp) if self.mtp is not None else 0
        return n_params(self) - skipped - mtp

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 0.0, gen: torch.Generator | None = None):
        for _ in range(n):
            logits = self(idx[:, -self.cfg.block_size :])[0][:, -1]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=gen)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
