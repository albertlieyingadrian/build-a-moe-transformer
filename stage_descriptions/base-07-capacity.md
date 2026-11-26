In this stage, you'll give every expert a queue limit, its **capacity**, and count the tokens turned away.

### A food stall with a fixed number of seats

When experts sit on different devices, each device has to reserve memory for its batch before the tokens arrive.
So each expert takes at most

```
capacity = ceil( capacity_factor × N × top_k / E )
```

token slots, where `N × top_k / E` is its fair share. A factor of 1.0 means "exactly the fair share"; 1.25 leaves a
25% margin. Slots past the limit are **dropped**: first come, first served, in token order. A dropped token gets
nothing from that expert. If every one of its slots is dropped, the MoE layer outputs zero for it, and the residual
connection (`x = x + ffn(x)`) carries it on to the next layer unchanged. It is not lost, just unprocessed.

The Switch Transformer and GShard train with a capacity factor; DeepSeek-V3 trains without dropping (its balancing is
good enough), and so do our training runs. This stage makes the cost of bad balancing visible: the more lopsided the
router, the more tokens a capacity limit throws away.

### Worked example

Eight experts, top-1, 64 tokens, and a router that starts with a lean towards expert 0 (`--skew 1.0` adds 1.0 to its
score):

```bash
$ ./your_program.sh capacity
capacity per expert: 8
wanted: [45, 3, 5, 0, 3, 3, 3, 2]
kept:   [8, 3, 5, 0, 3, 3, 3, 2]
dropped slots: 37 of 64 (57.8%)
tokens with every slot dropped: 37
their MoE output (the residual carries them on unchanged): 0.0
```

Expert 0 wanted 45 tokens and could seat 8, so 37 tokens skip this layer's FFN entirely. With
`--capacity-factor 0` (no limit) nothing is dropped, but expert 0 does 45 of the 64 tokens' work while expert 3 sits
idle.

### Tests

The tests check the capacity formula, that `kept = min(wanted, capacity)` for every expert, that some tokens are
dropped at factor 1.0, that their output is exactly zero, that nothing is dropped without a limit, and that a bigger
factor never drops more.

### Check yourself

1. With perfectly even routing and capacity factor 1.0, how many slots are dropped? With 1.0 and random routing?
2. Dropping "first come, first served" means tokens late in the batch are dropped more. What would be a fairer rule?
   (Some systems drop the slots with the lowest router weight first.)
