In this stage, you'll split every expert into smaller ones, DeepSeekMoE's **fine-grained experts**, and count what
that buys.

### More, smaller specialists

Take 8 experts of width 64, top-2. Cut each into 4 experts of width 16, and route to 8 of the 32. The weights are
the same. A token still uses the same number of weights. What changes is how many different teams of experts a token
can pick:

```
C(8, 2)  = 28 possible teams
C(32, 8) = 10,518,300 possible teams
```

A coarse expert has to be a generalist, because only 28 combinations must cover every kind of token. Fine-grained
experts can each hold a narrower skill and be combined freely, like hiring by skill rather than by department.
DeepSeekMoE splits experts this way (and DeepSeek-V3 has 256 routed experts, top-8); OLMoE uses 64 experts, top-8.

The shared expert is split the same way (1 of width 64 becomes 4 of width 16), so it keeps its size. The only part
that grows is the router: one row per expert.

### Worked example

```bash
$ ./your_program.sh fine-grained
split  experts  top_k  shared  hidden  expert weights  router  active params  combinations
    1        8      2       1      64          110592     512          37376  28
    4       32      8       4      16          110592    2048          38912  10518300
```

### The cost

Fine-grained experts are cheap in weights and expensive in bookkeeping: four times as many small matrix multiplies,
four times as many routing decisions to balance. Our loop over experts runs 32 small batches instead of 8, so it is
slower on a CPU, and you will see that in stage 13's timings.

### Tests

The tests run `fine-grained --d 32 --h 64 --split 4` and check that the expert weights are equal, the router is 4
times larger, the active weights differ only by the router, and the combinations are `C(8, 2)` and `C(32, 8)`.

### Check yourself

1. Why must top_k grow with the split? What would happen to active weights if you split but kept top-2?
2. More combinations sounds good. When could it hurt? (Hint: balancing 32 experts from a small batch.)
