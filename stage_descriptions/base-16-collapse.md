In this stage, you'll build a routing-collapse detector and watch both balancing methods rescue a router that
starts lopsided.

### What collapse looks like

A collapsed router sends most tokens to a few experts. Three numbers per layer catch it, all from the counts the MoE
layer already keeps:

```
max load   = busiest expert's share × E          1.0 even;  E / k = 4.0 is the worst top-2 allows
entropy    = H(shares) / log E                   1.0 even;  0.0 = one expert
dead       = experts with under a tenth of their fair share
```

The detector flags a layer when entropy < 0.75, max load > 2.0, or any expert is dead. On a made-up even layer
`[25, 25, 25, 25]` it says `ok`; on `[97, 1, 1, 1]` it raises all three flags.

### A router with a head start

Waiting for collapse to happen on its own makes a poor demonstration: at this scale it is slow and varies by seed.
So we cause it. `--skew 3` adds a fixed 3.0 to expert 0's router score in every layer, the same for every token, for
the whole run. Expert 0 starts in every token's top-2, the way the first checkout lane to open in a shop collects the
whole queue. Four runs from the same seed:

| run | head start | balancing |
|---|---|---|
| `none` | no | none |
| `none-skew` | yes | none |
| `aux-skew` | yes | auxiliary loss (stage 08) |
| `bias-skew` | yes | selection-only bias (stage 09) |

The fixed head start is a stress test: the aux loss has to learn router weights that cancel it, while the bias can
cancel it directly.

```bash
$ ./your_program.sh collapse data/tinyshakespeare.txt --steps 1500
```

### What we found (3 seeds, 1,500 steps)

| run | final max load, worst layer, seeds 0 / 1 / 2 | val loss |
|---|---|---|
| `none` | 2.79 / 3.21 / 3.22 | 1.548 / 1.549 / 1.530 |
| `none-skew` | 4.00 / 4.00 / 4.00 | 1.532 / 1.573 / 1.541 |
| `aux-skew` | 1.20 / 1.17 / 1.16 | 1.522 / 1.527 / 1.525 |
| `bias-skew` | 1.33 / 1.62 / 1.24 | 1.526 / 1.571 / 1.531 |

- **With a head start and no balancing, expert 0 never lets go**: max load 4.00, the most top-2 allows, at every
  evaluation of every seed.
- **Both methods recover within 150 steps** (the first evaluation after the start). Here the auxiliary loss ends a
  little more even (1.16-1.20) than the bias (1.24-1.62): the bias moves in fixed steps of 0.01 and keeps
  overshooting, the thermostat behaviour of stage 09.
- **Collapse happens on its own, too.** The `none` run has no head start, and its worst layer still ends at max load
  2.8-3.2 with dead experts. In every run the detector raises a flag at the first evaluation, before any training:
  a fresh router is already lopsided, because characters are (a space is about 15% of the text).
- Loss barely notices. The collapsed runs are within 0.05 of the balanced ones: on one CPU an idle expert costs
  nothing, which is exactly why collapse is easy to miss without a detector.

### Tests

The tests check both detector lines, then run 200 steps of each run on a tiny corpus and check that `none-skew`
stays above max load 3.0 and is flagged from step 0, that `bias-skew` ends below 2.0, and that both balancing methods
end below `none-skew`.

### Check yourself

1. Why can't the language-modelling loss alone undo the head start? (Hint: what does expert 0 do for the loss?)
2. Max load, entropy and dead experts can disagree. Make up a load vector that passes one check and fails another.
