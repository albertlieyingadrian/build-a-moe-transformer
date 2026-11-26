In this stage, you'll add DeepSeek-V3's **multi-token prediction** (MTP): a small module that, at every position,
also predicts the token after next.

### Naming the move after next

A chess student who is asked to name the next move and the one after it has to plan, not just react. MTP asks the
model the same: at position t the main model predicts token t+1, and one more block predicts token t+2:

```
h_t (main model's final state at t) ──► RMSNorm ──┐
                                                   concat ──► linear ──► one more block ──► shared head ──► token t+2
emb(token t+1)                      ──► RMSNorm ──┘
loss = main loss + 0.3 × MTP loss
```

The module is fed the true token t+1 during training, the way DeepSeek-V3 does, so it learns "given what the model
knows at t and the next token, what comes after?". It shares the embedding and the output head with the main model.
At inference it can be thrown away (the main model is unchanged in size), or used as a **draft**: guess t+2 cheaply,
and keep the guess when the main model agrees. That agreement rate is the draft acceptance rate of speculative
decoding.

```bash
$ ./your_program.sh mtp data/tinyshakespeare.txt --steps 3000
```

The command trains the same MoE with and without MTP and prints both main-model validation losses, the MTP module's
loss and top-1 accuracy on token t+2, and how often its guess matches the main model's own next-step guess.

### What we found (3 seeds, 3,000 steps)

| seed | main val loss, without MTP | with MTP | MTP loss on t+2 | MTP accuracy on t+2 | agrees with the main model |
|---|---|---|---|---|---|
| 0 | 1.524 | 1.519 | 1.523 | 0.553 | 0.781 |
| 1 | 1.518 | 1.514 | 1.490 | 0.560 | 0.791 |
| 2 | 1.539 | 1.512 | 1.503 | 0.557 | 0.783 |

- **MTP did not hurt the main model** and was a little better in all three seeds (-0.004 to -0.027), a gap the size
  of the seed noise.
- **As a draft it would be accepted about 78-79% of the time**: its guess for t+2 matches what the main model itself
  would predict at t+1, given the true token t+1. That is the number that decides how much speculative decoding saves.
- The MTP module has 236,064 weights (a dense block and a mixing matrix), used only in training here.

### Tests

The tests check that the MTP loss falls well below guessing, that the active parameters are unchanged, that the
agreement is a rate between 0 and 1, and that the main loss moves by less than 0.3.

### Check yourself

1. Why can the MTP loss on token t+2 be lower than the main loss on token t+1?
2. DeepSeek-V3 reports 85-90% acceptance for its MTP draft on real text. What makes acceptance easier on Tiny
   Shakespeare's characters, and what makes it harder?
