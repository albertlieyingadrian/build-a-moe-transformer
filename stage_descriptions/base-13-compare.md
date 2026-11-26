In this stage, you'll give the MoE a fair fight: train it against dense models with the same active parameters and
with the same total parameters, on the same text, steps, optimiser and seed.

### Two fair baselines

An MoE can be compared with a dense model in two ways, and both matter:

- **Same active parameters** (`dense`): one SwiGLU FFN of hidden width `(top_k + n_shared) × expert_hidden = 384`.
  Each token does the same work as in the MoE. Does storing more weights, and routing between them, buy anything?
- **Same total parameters** (`dense-wide`): one FFN of hidden width `(n_routed + n_shared) × expert_hidden = 1152`.
  The MoE's total weights, all used by every token: three times the work per token. This is the ceiling the MoE is
  chasing: how much of a dense model's quality does it keep with a third of the compute?

And four MoEs with the same shape (4 layers, width 128, MLA, 8 routed experts of hidden width 128, top-2, 1 shared),
differing in one thing each:

| variant | balancing | experts |
|---|---|---|
| `moe-none` | none | 8 × 128, top-2 |
| `moe-aux` | auxiliary loss, coefficient 0.01 (stage 08) | 8 × 128, top-2 |
| `moe-bias` | selection-only bias, rate 0.01 (stage 09) | 8 × 128, top-2 |
| `moe-bias-fine` | bias | 32 × 32, top-8, 4 shared of 32 (stage 11) |

### The `compare` command

```bash
$ ./your_program.sh compare data/tinyshakespeare.txt --steps 3000 \
      --variants dense,dense-wide,moe-none,moe-aux,moe-bias,moe-bias-fine
```

It trains each variant, prints a table (total parameters, active parameters, final validation loss, mean max load
over layers), saves `compare_s<seed>.json`, one checkpoint per variant and `compare_s<seed>.png` (validation loss
over training).

### What we found (3 seeds)

| variant | total params | active params | val loss, seeds 0 / 1 / 2 | mean ± sd | max load |
|---|---|---|---|---|---|
| `dense` | 828,928 | 828,928 | 1.512 / 1.525 / 1.550 | 1.529 ± 0.019 | - |
| `dense-wide` | 2,008,576 | 2,008,576 | 1.531 / 1.493 / 1.570 | 1.531 ± 0.038 | - |
| `moe-none` | 2,012,672 | 833,024 | 1.513 / 1.529 / 1.531 | 1.524 ± 0.010 | 2.55 / 2.29 / 2.90 |
| `moe-aux` | 2,012,672 | 833,024 | 1.516 / 1.536 / 1.539 | 1.530 ± 0.013 | 1.10 / 1.06 / 1.15 |
| `moe-bias` | 2,012,672 | 833,024 | 1.524 / 1.518 / 1.539 | 1.527 ± 0.011 | 1.21 / 1.23 / 1.32 |
| `moe-bias-fine` | 2,024,960 | 845,312 | 1.509 / 1.528 / 1.520 | 1.519 ± 0.009 | 2.65 / 2.33 / 2.36 |

- **Faster, then tied.** Figure `assets/compare.png` (right) plots each variant's validation loss minus `dense`'s,
  on the same seed and the same validation windows. From step 300 to step 1,800 every MoE is ahead in every seed, by
  0.03-0.05 on average. Then the gap closes: at step 3,000 it is -0.010 to +0.001, inside the seed-to-seed spread.
  `dense-wide`, with 2.4 times the weights all in use, follows the same path.

  | step | `moe-none` | `moe-aux` | `moe-bias` | `moe-bias-fine` | `dense-wide` |
  |---|---|---|---|---|---|
  | 600 | -0.033 | -0.047 | -0.040 | -0.048 | -0.024 |
  | 1,200 | -0.034 | -0.047 | -0.041 | -0.037 | -0.027 |
  | 1,800 | -0.025 | -0.026 | -0.031 | -0.025 | -0.015 |
  | 2,400 | -0.010 | -0.008 | -0.014 | -0.018 | -0.008 |
  | 3,000 | -0.005 | +0.001 | -0.002 | -0.010 | +0.002 |

  (Mean over 3 seeds of the loss minus `dense`'s; below zero is better. Mid-training evaluations use 10 batches,
  the final one 40.)
- **Why the gap closes:** Tiny Shakespeare is about a million characters and 3,000 steps of 32 × 128 show the model
  about 12 million, so every model has read the text a dozen times. Extra stored weights help while there is still
  something new to fit and stop helping when there isn't. MoE gains in the literature come from large, varied data,
  where that point is never reached.
- **Balancing is free here.** The unbalanced MoE is as good as the balanced ones on loss, while its busiest expert
  does 2.3-2.9 times its share of the work. On real hardware that expert's device sets the pace for everyone.
- **Time:** on a CPU the MoE takes about 1.1-1.3 times as long per step as `dense` (802-996 s against 638-800 s), and
  the 32-expert version about twice as long: the Python loop over experts does many small matrix multiplies.

### Tests

The tests train `dense`, `moe-bias` and `dense-wide` for 250 steps on a tiny corpus and check that the active
parameters of `dense` and `moe-bias` match within 3%, that `dense-wide` matches the MoE's total within 3%, that all
three learn, that the MoE is no worse than `dense` plus a 0.15 tolerance, and that its max load stays under 2.

### Check yourself

1. Which baseline would a company paying for inference care about, and which would someone fitting a model on a
   phone care about?
2. `moe-bias-fine` has the same weights as `moe-bias` and more possible teams. Why might it not win at this scale?
