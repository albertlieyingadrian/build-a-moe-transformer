In this stage, you'll open the trained MoE and measure which characters go to which expert, and whether that is more
than chance.

### Three questions about routing

1. **Is the load even?** The share of top-k slots each expert gets, per layer.
2. **Does the token decide the expert?** The mutual information between the input character and its top-1 expert,
   divided by the entropy of the expert: `MI(token; expert) / H(expert)`. Zero means the router ignores which
   character it sees; one means the character alone fixes the expert.
3. **What does each expert get?** For every expert, the characters most over-represented among its tokens (share at
   that expert ÷ share overall), and a table of six character classes (space, newline, vowel, consonant, upper,
   punctuation) × experts, drawn as a heatmap.

### Two controls

- **Shuffled labels.** Keep the token list and the expert list, shuffle the experts across positions, and recompute
  the information. That keeps both distributions and breaks the link, so it should be close to zero. It measures the
  estimator's own bias.
- **An untrained model.** The same architecture with fresh random weights. A router is a matrix on the residual
  stream, and at the first layer that stream is mostly the token embedding, so even random routing can depend on the
  token. The untrained number tells you how much of the "specialisation" training added.

```bash
$ ./your_program.sh route-lab data/tinyshakespeare.txt --ckpt runs/seed0/moe-bias_s0.pt
```

### What we found (`moe-bias`, 3 seeds)

| layer | trained, seeds 0 / 1 / 2 | untrained model | shuffled control |
|---|---|---|---|
| 0 | 0.96 / 0.94 / 0.86 | 0.83 / 0.79 / 0.76 | 0.001 |
| 1 | 0.51 / 0.43 / 0.60 | 0.63 / 0.66 / 0.66 | 0.001 |
| 2 | 0.47 / 0.54 / 0.43 | 0.47 / 0.56 / 0.54 | 0.001 |
| 3 | 0.60 / 0.59 / 0.58 | 0.37 / 0.43 / 0.40 | 0.001 |

- **Layer 0 routes by character, and almost entirely so** (0.86-0.96 of the expert's entropy). But an untrained model
  already scores 0.76-0.83: at the first layer a router is a matrix applied to a token embedding, so its choice is a
  function of the token from the start. Training sharpens it rather than creating it.
- **Later layers route on more than the current character** (0.43-0.60), and in layers 1-2 training leaves the
  number where the untrained model had it. Only the last layer is clearly more token-specific after training.
- **The heatmap** (`routing_heatmap.png`, seed 0, in the README) shows whole classes claimed by one expert in layer
  0: half of all spaces' slots go to expert 3, half of all newlines' to expert 0. Later layers still split spaces,
  newlines, capitals and punctuation onto their own experts, while vowels and consonants spread out.
- The shuffled control is 0.001 in every layer: the estimator's bias is negligible with about 80,000 tokens.

### Tests

The tests train a small MoE (300 steps) and check that the shuffled control is under 0.05, that the trained
model's information is at least 0.1 above it in every layer, that the class table's rows sum to 1, and that the
heatmap is written.

### Check yourself

1. Why compare with an untrained model and not only with shuffled labels?
2. Routing by the current character alone would be a lookup table. What in the residual stream could let a later
   layer route by context instead?
