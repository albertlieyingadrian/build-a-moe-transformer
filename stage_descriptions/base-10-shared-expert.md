In this stage, you'll add a **shared expert**: one expert every token uses, with no routing at all.

### The common knowledge desk

Every routed expert needs some of the same basics: how spaces and capitals work, the most common letter pairs. If
each expert must relearn them, that is the same knowledge stored E times. DeepSeekMoE's fix is to put a desk in the
middle of the food court that serves every order. The shared expert learns what everyone needs, and the routed
experts are free to specialise.

```
moe(x) = shared(x) + Σ over the k chosen experts  w_e · expert_e(x)
```

Several shared experts that all see every token add up to one wider expert, so we store `n_shared` of them as a single
expert of width `n_shared × h`.

### Counting active weights

A token uses the shared expert, its k routed experts, and the router:

```
active FFN weights per token = (k + n_shared) · 3·d·h + d·E
total FFN weights            = (E + n_shared) · 3·d·h + d·E
```

### Worked example

```bash
$ ./your_program.sh shared-expert
parameters: routed 98304, shared 12288, router 512, total 111104
active per token: 2 routed + 1 shared + router = 37376
routed experts off: the layer is the shared expert: True
layer = shared part + routed part: True
```

Eight routed experts plus one shared, top-2: the layer stores 111,104 weights and each token uses 37,376 of them, a
third. A dense FFN of width `(2 + 1) × 64 = 192` has 36,864 weights: the same as the MoE's active count, minus the
router. That is how stage 13 builds a fair dense baseline.

### Tests

The tests run `shared-expert --d 32 --h 16 --experts 6` and check every count, and that switching off all routed
experts leaves exactly the shared expert.

### Check yourself

1. Ablating the shared expert should hurt more than ablating one routed expert (stage 15 measures it). Why?
2. DeepSeek-V3 has 256 routed experts, top-8 and 1 shared. What fraction of its expert weights does one token use?
