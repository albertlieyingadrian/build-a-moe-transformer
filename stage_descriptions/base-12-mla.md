In this stage, you'll replace attention with DeepSeek's **Multi-head Latent Attention** (MLA), with decoupled RoPE,
and measure how much smaller its cache is.

### The cache problem

To generate text fast, a model keeps every past token's keys and values, the **KV cache**, so it does not recompute
them for every new token. Plain multi-head attention caches a key and a value of `n_embd` numbers each: `2 × n_embd`
floats per token per layer. For long contexts that cache, not the weights, fills the GPU.

### Zip the cache

MLA caches one small **latent** vector `c_kv` per token (`kv_rank` numbers) and rebuilds every head's key and value
from it with two up-projections. Like storing a zip file instead of the folder:

```
x ──► W_dkv ──► c_kv (kv_rank) ──► RMSNorm ──┬─► W_uk ──► content keys   (heads × head_dim)
                 [cached]                     └─► W_uv ──► values         (heads × head_dim)
```

At inference the up-projections can be folded into the query and output matrices, so the model never unzips.

### Why RoPE needs its own slice

RoPE (rotary position embedding) rotates queries and keys by an angle that grows with position. The score between
positions i and j then depends only on `i - j`: the model sees distance, not absolute place. But a rotation that
depends on position, sitting between `c_kv` and the query, would stop `W_uk` being folded into the query. So MLA
**decouples** position: each head's query gets `rope_dim` extra numbers, all heads share one `rope_dim`-wide key
(also cached), RoPE is applied only to those, and

```
score = q_content · k_content  +  q_rope · k_rope
```

Our model drops the learned position table when it uses MLA: position now lives entirely in RoPE.

### Worked example

Width 64, 4 heads, `kv_rank = 16`, `rope_dim = 8`:

```bash
$ ./your_program.sh mla
cache floats per token per layer: MHA 128, MLA 24 (5.3x smaller)
parameters: MHA 16384, MLA 13840
MLA weight on future positions: 0.0
change at earlier positions after editing the last one: 0.0
RoPE score depends only on the distance: True
```

The last line places a query at 7 and a key at 3, then at 12 and 8: the RoPE score is the same, because the distance
is the same. DeepSeek-V2 reports a 93% smaller KV cache than its dense predecessor, with 512 latent numbers and 64
RoPE numbers per token against 128 heads of 128.

### Tests

The tests run `mla --d 96 --kv-rank 24 --rope-dim 12` and check the cache sizes (192 vs 36), the causal mask, the
edit test and the RoPE distance test.

### Check yourself

1. Why is the RoPE key shared across heads? How big would the cache be if every head had its own?
2. A smaller cache with a smaller `kv_rank` sounds free. What does the model lose? (Stage 13 trains with
   `kv_rank = n_embd / 4`.)
