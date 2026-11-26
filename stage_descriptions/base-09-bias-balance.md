In this stage, you'll balance the load the DeepSeek-V3 way: no extra loss, just a bias per expert that steers the
choice and nothing else.

### A thermostat, not a penalty

Give each expert a number `b_i`, start at 0. Add it to the scores **only when choosing** the top-k; the weights still
come from the plain probabilities:

```
chosen  = top_k of (probs + b)          the bias decides who gets picked
weights = probs[chosen] / their sum     the bias never touches the weights
after every step:  b_i += rate · sign(mean load - load_i)
```

An overloaded expert's bias goes down by `rate`, an underloaded one's goes up. Like a thermostat it doesn't care how
far off the temperature is, only which way. Because the bias is a buffer, not a trained parameter, no gradient flows
into it, and the language-modelling loss is left alone: nothing competes with it, unlike the aux loss of stage 08.

DeepSeek-V3 uses `rate = 0.001` over hundreds of thousands of steps. Our runs are a few thousand steps long, so we use
0.01.

### Worked example

A router that never trains, with expert 0 favoured, sees a fresh batch of 512 random tokens every step:

```bash
$ ./your_program.sh bias-balance
step  max load  entropy  bias[0]
   0      2.05    0.968  +0.000
  40      1.19    0.995  -0.310
  80      1.35    0.994  -0.240
 120      1.27    0.996  -0.220
 160      1.18    0.998  -0.180
 200      1.12    0.997  -0.150
max load: 2.05 -> 1.12
gate weights ignore the bias: True
bias is trained by gradients: False
```

`max load` is the busiest expert's share times E: 1.0 is perfectly even and 4.0 (E / k) is the worst top-2 allows.
The bias of expert 0 overshoots to -0.31, then settles back. Because each step moves every bias by the same fixed
amount, it hovers around the balance point instead of landing on it.

### Tests

The tests check that max load falls from above 1.6 to below 1.4, that the weights are the plain renormalised
probabilities, that the bias is not a parameter, and that with `--rate 0` nothing changes.

### Check yourself

1. Why use `sign(...)` rather than the size of the imbalance? What would a large imbalance do to a proportional
   update?
2. The bias can make the router pick an expert whose probability is not in the top-k. Is that a bug?
