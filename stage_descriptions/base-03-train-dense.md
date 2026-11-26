In this stage, you'll train the dense GPT and sample from it. This is the baseline every MoE result is compared with.

### The training loop

```
for step in range(steps):
    x, y = get_batch(train)            # random windows and their shifted targets
    loss = cross_entropy(model(x), y)
    loss.backward()                    # every weight learns which way to move
    clip the gradient norm to 1        # one bad batch can't throw the model off a cliff
    AdamW step                         # move, with per-weight step sizes
```

The learning rate warms up over the first 5% of steps and then follows a cosine down to a tenth of its peak. Every
`steps / 10` steps we measure the loss on validation text.

### The baseline we'll keep using

The default model is 4 layers of width 128 with a 128-character context. Its FFN has hidden width 384 (`3 × 128`):
exactly the expert weights one token touches in the MoE of stage 13 (2 routed experts and 1 shared expert of hidden
width 128). That makes it the **same-active-parameters** baseline.

```bash
$ ./your_program.sh train-dense data/tinyshakespeare.txt --steps 3000
parameters: 821760
step     0  val 4.174
...
final val loss: 1.528 (guessing: ln V = 4.174)
training time: 210 s
sample (greedy):
...
```

(The exact loss, time and sample of a committed run are in `runs/seed0/` after `./run_all.sh`.)

Greedy sampling always takes the most likely next character, which is deterministic and soon loops ("the the the").
`generate` also takes a temperature: divide the logits by it and sample.

### Tests

The tests train a tiny model for 200 steps on a short corpus and check that the validation loss ends below half of
`ln V`, that `dense.pt` is saved and that a sample is printed.

### Check yourself

1. Why does the training loss usually end lower than the validation loss?
2. Double the width to 256 and the loss drops. What does that cost per token? Stage 13 asks whether an MoE can buy
   the same drop for less.
