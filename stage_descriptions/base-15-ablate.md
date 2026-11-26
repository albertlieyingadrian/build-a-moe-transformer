In this stage, you'll switch experts off one at a time and measure whose predictions get worse.

### Taking a stall out of the food court

To ablate routed expert e in layer l, run the model as usual but skip e's contribution: the tokens that chose it keep
their other expert's answer (and the shared expert's), with the weights left as they were. Then measure the
next-character loss on the same validation windows as the unablated model, position by position, and split the
change into two groups:

- the positions whose token had e in its top-k ("its own tokens");
- every other position.

If experts really divide the work, switching one off should hurt its own tokens much more than the rest. The rest
can still be hurt a little, because a changed state at one position is read by later positions through attention.
The shared expert is ablated the same way, one layer at a time.

```bash
$ ./your_program.sh ablate data/tinyshakespeare.txt --ckpt runs/seed0/moe-bias_s0.pt
```

### What we found (`moe-bias`, 3 seeds, 80,000 positions)

| seed | one routed expert off | shared expert off | on its own tokens | on the rest | own > rest |
|---|---|---|---|---|---|
| 0 | +0.112 | +0.211 | +0.328 | +0.040 | 32 of 32 |
| 1 | +0.124 | +0.163 | +0.392 | +0.037 | 32 of 32 |
| 2 | +0.105 | +0.189 | +0.354 | +0.027 | 32 of 32 |

(Changes in validation loss, nats per character; the baseline is about 1.53.)

- **Experts divide the work.** Every one of the 96 ablations hurt the expert's own tokens more than the rest, by
  about nine times on average.
- **Layer 0 matters most.** Switching off a layer-0 expert costs +0.35 to +0.40 on average; a routed expert in layers
  1-3 costs +0.02 to +0.03. The first layer's experts do much of the per-character work, and later layers have
  redundancy: the other chosen expert and the shared expert cover for a missing one.
- **The shared expert costs more than a routed one** (+0.16 to +0.21 against +0.11 to +0.12), because every token
  uses it. Most of that is layer 0 too (+0.68 in seed 0, against +0.04 to +0.07 in later layers).

### Tests

The tests check one row per routed expert and per shared expert in each layer, that the mean change over routed
experts is positive, and that at least 12 of the 16 ablations hurt their own tokens more than the rest.

### Check yourself

1. Why measure on the same windows for every ablation?
2. An expert whose ablation barely changes the loss could be useless, or redundant with another. How would you tell
   which?
