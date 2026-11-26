# Build a Mixture-of-Experts Transformer

[![test](https://github.com/albertlieyingadrian/build-a-moe-transformer/actions/workflows/test.yml/badge.svg)](https://github.com/albertlieyingadrian/build-a-moe-transformer/actions/workflows/test.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertlieyingadrian/build-a-moe-transformer/blob/main/notebooks/quickstart.ipynb)

A dense transformer sends every token through the same feed-forward network. A **Mixture-of-Experts (MoE)**
transformer keeps many smaller feed-forward networks, the **experts**, and a **router** that sends each token to only
a few of them: more stored knowledge, the same work per token. DeepSeek-V3 stores 671B parameters and uses 37B per
token; Mixtral 8x7B stores 47B and uses 13B.

In this course you build a DeepSeek-style MoE from scratch, in plain PyTorch, on a laptop CPU:

1. a character tokenizer and a dense GPT baseline (stages 01-03);
2. the MoE layer piece by piece: a SwiGLU expert, a softmax top-k router, batched dispatch and combine, capacity and
   dropped tokens (stages 04-07);
3. two ways to keep the router from piling tokens onto a few experts: the Switch-style auxiliary loss and
   DeepSeek-V3's auxiliary-loss-free bias (stages 08-09);
4. DeepSeekMoE's shared expert and fine-grained experts, and DeepSeek-V2's Multi-head Latent Attention with
   decoupled RoPE (stages 10-12);
5. a fair fight: the MoE against dense models with the same active and the same total parameters (stage 13);
6. an interpretability lab: which characters go to which expert, what breaks when you switch one off, and how to
   detect routing collapse (stages 14-16);
7. one extension: DeepSeek-V3's multi-token prediction (stage 17).

The layout follows the stage-by-stage style of CodeCrafters' build-your-own-X courses (https://codecrafters.io): a
`course-definition.yml`, one description per stage in `stage_descriptions/`, a starter in `starter_templates/python/`,
the reference program after every stage in `solutions/python/`, and tests per stage. The model code follows the
structure of nanoGPT. Nothing is copied from either; both are credited below. This is an independent course, not
affiliated with or endorsed by CodeCrafters.

## What you'll build

```
                    dense FFN                                     MoE FFN (stages 04-11)

  x ──► RMSNorm ──► [ one SwiGLU, hidden 384 ] ──► + x      x ──► RMSNorm ──► router ──► top-2 of 8 experts (hidden 128)
                                                                     │                   + 1 shared expert (hidden 128)
                                                                     └── bias steers the choice (stage 09)  ──► + x
```

By stage 17 the program trains a 4-layer, width-128 character model with Multi-head Latent Attention and an MoE in
every layer (8 routed experts, top-2, 1 shared, 2.0M weights, 0.83M active per token), compares it with two dense
models, and draws where its tokens go:

![Validation loss of six models after 3,000 steps, three seeds](assets/compare.png)

![Share of each character class's routing slots per expert, four layers](assets/routing_heatmap.png)

![Max load of the worst layer over training, with expert 0 given a head start](assets/collapse.png)

## What you'll find

Three seeds, every number from `./run_all.sh` on 8-vCPU cloud CPU containers (4 threads each; 11-26 minutes per
model) and printed by `python scripts/summarize.py`. Tiny Shakespeare, 3,000 steps of 32 windows of 128 characters,
4 layers of width 128 with latent attention.

| model | total params | active params | val loss, seeds 0 / 1 / 2 | mean ± sd | max load |
|---|---|---|---|---|---|
| dense, same active parameters | 828,928 | 828,928 | 1.512 / 1.525 / 1.550 | 1.529 ± 0.019 | - |
| dense-wide, same total parameters | 2,008,576 | 2,008,576 | 1.531 / 1.493 / 1.570 | 1.531 ± 0.038 | - |
| MoE, no balancing | 2,012,672 | 833,024 | 1.513 / 1.529 / 1.531 | 1.524 ± 0.010 | 2.55 / 2.29 / 2.90 |
| MoE, auxiliary loss | 2,012,672 | 833,024 | 1.516 / 1.536 / 1.539 | 1.530 ± 0.013 | 1.10 / 1.06 / 1.15 |
| MoE, auxiliary-loss-free bias | 2,012,672 | 833,024 | 1.524 / 1.518 / 1.539 | 1.527 ± 0.011 | 1.21 / 1.23 / 1.32 |
| MoE, fine-grained (32 experts, top-8), bias | 2,024,960 | 845,312 | 1.509 / 1.528 / 1.520 | 1.519 ± 0.009 | 2.65 / 2.33 / 2.36 |

"Max load" is the busiest expert's share of the routing slots times the number of experts, averaged over layers:
1.0 is perfectly even, 4.0 the worst top-2 allows.

- **The MoE learns faster, then the data runs out.** Paired with the dense model of the same active parameters (same
  seed, same validation text), every MoE variant is ahead in every seed from step 300 to step 1,800, by 0.03-0.05 nats
  per character on average (worst seed still ahead by 0.003-0.040). By step 3,000 the gap has closed to a tie (-0.010
  to +0.001 on average, inside the 0.01-0.04 seed-to-seed spread). Tiny Shakespeare is 1 MB, and by step 3,000 every
  model has read it about a dozen times, so extra stored weights have little left to store; the dense model with the
  MoE's total weights (and three times the work per token) ends tied too. The one-file version at 8-19M parameters
  on an H100 finds the same tie at its best validation loss.
- **Both balancing methods work, at no cost in loss.** With no balancing the busiest expert carries 2.3-2.9 times its
  fair share; the auxiliary loss brings that to 1.06-1.15 and the bias to 1.21-1.32. Fine-grained experts with the
  same bias rate stay uneven (2.3-2.7): 32 experts would need a faster rate or more steps.
- **A router collapses on its own.** Without balancing, the worst layer ends with one or two dead experts (under a
  tenth of their fair share) in every seed, and the stage-16 detector flags it from the first evaluation.
- **Experts divide the work.** Switching off one routed expert hurt the tokens it served more than the rest in 32 of
  32 ablations in every seed (+0.33 to +0.39 nats against +0.03 to +0.04). Switching off the shared expert costs more
  than a routed one (+0.16 to +0.21 against +0.11 to +0.12), and layer 0's experts matter most.
- **Routing follows the character, and most of that is there before training.** In layer 0 the character carries
  0.86-0.96 of the information about its expert (shuffled control 0.001), but an untrained model already scores
  0.76-0.83: a router reading the token embedding is a lookup on the token from the start. In later layers training
  adds little or nothing (0.43-0.60 against 0.37-0.66). Spaces and punctuation get experts of their own.
- **Recovery from a lopsided start.** Give expert 0 a fixed head start: with no balancing it keeps it (max load 4.00
  in all seeds); the auxiliary loss pulls it back to 1.16-1.20 within 150 steps, the bias to 1.24-1.62.
- **Multi-token prediction** (extension) agrees with the main model's own next guess 78-79% of the time, a draft
  acceptance rate, and leaves the main loss within 0.03 (1.519 / 1.514 / 1.512 with it, 1.524 / 1.518 / 1.539
  without).

## Limitations

- Toy scale: 0.8-2.0M parameters, character-level Tiny Shakespeare, 3,000 steps. The findings describe this model and
  this data; they are not claims about large MoE language models, where the same comparisons are run at billions of
  parameters and tokens.
- Three seeds per result. Differences smaller than the seed-to-seed spread in the tables are noise.
- Our dispatch is a Python loop over experts on one device: clear, not fast. Real systems use grouped matrix
  multiplies and expert parallelism across devices, where load balance turns directly into wall-clock time.
- The router uses a softmax, as DeepSeek-V2 does; DeepSeek-V3's sigmoid scores and its small sequence-level balance
  loss are described in stages 05 and 08 but not implemented. Training never drops tokens (capacity factor 0); stage
  07 shows dropping on its own.
- The collapse experiments (stage 16) start the router lopsided on purpose, with a fixed head start for one expert;
  they show detection and recovery, not how often collapse happens on its own.
- The interpretability stages use simple statistics and single ablations with a shuffled-label control; they show
  methods, not a full mechanistic account of what each expert computes.

## Stages

| # | Stage | Command | What you learn |
|---|---|---|---|
| 01 | [Characters to numbers](stage_descriptions/base-01-tokenize.md) | `tokenize` | a character vocabulary, a train/validation split, batches |
| 02 | [A dense GPT](stage_descriptions/base-02-forward.md) | `forward` | RMSNorm, causal attention, the loss of a model that knows nothing (`ln V`) |
| 03 | [Train the dense baseline](stage_descriptions/base-03-train-dense.md) | `train-dense` | the training loop, validation loss, sampling |
| 04 | [The SwiGLU expert](stage_descriptions/base-04-expert.md) | `expert` | gated feed-forward networks, `3·d·h` weights |
| 05 | [The router: softmax and top-k](stage_descriptions/base-05-router.md) | `router` | scoring experts, keeping k, renormalised weights |
| 06 | [Dispatch and combine](stage_descriptions/base-06-dispatch.md) | `dispatch` | batching tokens by expert, `index_add_`, a token-by-token reference |
| 07 | [Capacity and dropped tokens](stage_descriptions/base-07-capacity.md) | `capacity` | capacity factor, dropped slots, the residual as a safety net |
| 08 | [The load-balancing loss](stage_descriptions/base-08-aux-loss.md) | `aux-loss` | `E · Σ f·P`: 1 when even, `E / k` when collapsed |
| 09 | [Auxiliary-loss-free balancing](stage_descriptions/base-09-bias-balance.md) | `bias-balance` | DeepSeek-V3's selection-only bias |
| 10 | [The shared expert](stage_descriptions/base-10-shared-expert.md) | `shared-expert` | always-on experts, active vs total parameters |
| 11 | [Fine-grained experts](stage_descriptions/base-11-fine-grained.md) | `fine-grained` | same weights, `C(32, 8)` instead of `C(8, 2)` teams |
| 12 | [Multi-head Latent Attention](stage_descriptions/base-12-mla.md) | `mla` | a compressed KV cache, decoupled RoPE |
| 13 | [MoE vs dense](stage_descriptions/base-13-compare.md) | `compare` | matched active and matched total parameters |
| 14 | [Who goes where](stage_descriptions/base-14-route-lab.md) | `route-lab` | routing tables, token → expert information, a shuffled control |
| 15 | [Switching experts off](stage_descriptions/base-15-ablate.md) | `ablate` | per-expert ablation, targeted damage, the shared expert |
| 16 | [Routing collapse](stage_descriptions/base-16-collapse.md) | `collapse` | a collapse detector; recovery with the aux loss and the bias |
| 17 | [Multi-token prediction](stage_descriptions/ext-17-mtp.md) (extension) | `mtp` | predicting the token after next; a draft acceptance rate |

## How to run

Requirements: Python 3.10+, PyTorch 2.x (`pip install torch`), and `pip install -r requirements.txt` (numpy,
matplotlib, pytest, ruff). Everything runs on a CPU. The only download is the 1 MB Tiny Shakespeare file (`make
data`).

```bash
cp -r starter_templates/python/code my-moe          # your working copy; start with app/main.py
make test STAGE=01 DIR=my-moe                        # one stage's tests against your code
make test-upto STAGE=08 DIR=my-moe                   # every stage so far
make test-solutions                                  # every reference solution against its stage (what CI runs)
make run-all                                         # every number quoted in the README and stage descriptions
make solutions                                       # regenerate solutions/python/01-16 from 17
```

Stuck? `solutions/python/NN-<slug>/code/` is the program right after stage NN, and `run_all.sh` lists every command.
The whole test suite takes under a minute on a laptop.

## Further reading and code

What each one is, and what this course borrows. We wrote our own code; nothing below is copied.

**The structure of this course and its code**

- nanoGPT, https://github.com/karpathy/nanoGPT: the shape of the model and the training loop (stages 01-03).
- CodeCrafters' open-source course repos (MIT), https://github.com/codecrafters-io/build-your-own-tokenizer and
  https://github.com/codecrafters-io/build-your-own-inference-engine: the course layout.
- nanowhale, https://github.com/huggingface/nanowhale: a ~110M DeepSeek-V4-style model trained from scratch, with MLA,
  an MoE of 4 routed experts and 1 shared expert (top-2), Hyper-Connections and multi-token prediction. It has no
  licence, so we read it as prior art and took no code from it. It showed that a from-scratch DeepSeek-style model
  is small enough to teach.
- build-a-looped-transformer, https://github.com/albertlieyingadrian/build-a-looped-transformer: the sister course
  this one copies its own layout and testing approach from.

**Mixture of Experts**

- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017),
  https://arxiv.org/abs/1701.06538: the top-k gated MoE layer and an importance loss to balance it.
- GShard (2020), https://arxiv.org/abs/2006.16668: top-2 routing, expert capacity and an auxiliary loss at scale.
- Switch Transformers (2021), https://arxiv.org/abs/2101.03961: top-1 routing, the capacity factor (stage 07) and the
  `E · Σ f·P` load-balancing loss (stage 08).
- DeepSeekMoE (2024), https://arxiv.org/abs/2401.06066: fine-grained experts (stage 11) and shared experts (stage 10).
- DeepSeek-V2 (2024), https://arxiv.org/abs/2405.04434: Multi-head Latent Attention with decoupled RoPE (stage 12),
  device- and expert-level balance losses.
- Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts (2024), https://arxiv.org/abs/2408.15664: the
  selection-only bias and its sign update (stage 09).
- DeepSeek-V3 (2024), https://arxiv.org/abs/2412.19437: the bias at 671B scale, sigmoid router scores, and
  multi-token prediction (stage 17).
- OLMoE (2024), https://arxiv.org/abs/2409.02060: a fully open MoE (64 experts, top-8) with analyses of routing
  saturation, expert co-activation and specialisation by token and domain (stages 14-16 run small versions).
- Mixtral of Experts (2024), https://arxiv.org/abs/2401.04088: 8 experts, top-2, and a routing analysis that finds
  experts follow syntax more than topic.
- ST-MoE (2022), https://arxiv.org/abs/2202.08906: the router z-loss and a study of what experts specialise in.
- RoFormer (2021), https://arxiv.org/abs/2104.09864: rotary position embedding.
- GLU Variants Improve Transformer (2020), https://arxiv.org/abs/2002.05202: SwiGLU (stage 04).
- Root Mean Square Layer Normalization (2019), https://arxiv.org/abs/1910.07467: RMSNorm (stage 02).
- Tiny Shakespeare comes from the char-rnn repo, https://github.com/karpathy/char-rnn.
- nanomoe-transformer, https://github.com/albertlieyingadrian/nanomoe-transformer: the same model in one file of
  PyTorch, with a dense vs MoE comparison and a routing heatmap.

## License

MIT. See [LICENSE](LICENSE).
