In this stage, you'll build the dense baseline that every MoE in this course is measured against: a small GPT.

### The shape of a GPT

```
ids ──► token embedding (+ position embedding) ──► x        (one vector of n_embd numbers per position)
         Block × n_layer:   x = x + Attention(RMSNorm(x))   "look at earlier positions"
                            x = x + FFN(RMSNorm(x))         "think about what you saw"
        RMSNorm ──► head ──► logits                          (one score per possible next character)
```

`x` is the **residual stream**: a whiteboard every block reads and adds a note to. Attention is the only place where
positions talk to each other. The FFN (feed-forward network) works on each position alone; it is where most of a
transformer's weights live, and it is the part an MoE replaces.

Three pieces, each a few lines:

- **RMSNorm** rescales each vector so its root-mean-square is 1, then multiplies by a learned gain. It is LayerNorm
  without subtracting the mean and without a bias; DeepSeek and Llama use it.
- **Causal attention**: every position writes a query, a key and a value; it reads the values of earlier positions
  whose keys match its query. A mask sets the scores for later positions to minus infinity, so no one reads the
  future.
- **The FFN**: for now, one SwiGLU network (stage 04 explains it) of hidden width `3 × expert_hidden`. That width is
  chosen to match, weight for weight, what one token will use in the MoE (two routed experts plus one shared).

### What a model that knows nothing scores

The loss is the cross-entropy of the right next character: `-ln p(right one)`. A fresh model spreads its bets about
evenly over the 65 characters, so `p ≈ 1/65` and the loss is about `ln 65 = 4.174`. If your fresh model scores much
higher, its initial weights are too big.

### Worked example

```bash
$ ./your_program.sh forward data/tinyshakespeare.txt "First Citizen"
ids shape: (1, 13)
logits shape: (1, 13, 65)
parameters: 119232
loss at initialisation: 4.170
ln(vocab size): 4.174
change at earlier positions after editing the last token: 0.0
```

The last line is the real test of the mask: change the last character, and the predictions at every earlier position
must not move at all.

### Tests

The tests check the shapes, a loss within 0.2 of `ln V`, and the `0.0` on the last line.

### Check yourself

1. Why initialise weights with a small standard deviation (0.02)? What would the loss at initialisation be if every
   logit were huge and random?
2. Attention on its own has no idea of order. Where does order come from here? (Stage 12 changes the answer.)
