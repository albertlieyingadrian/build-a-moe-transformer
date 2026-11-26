In this stage, you'll add the classic fix for lopsided routing: an **auxiliary loss** that pushes the router to
spread its tokens.

### Why routers collapse

Routing feeds itself. An expert that gets more tokens early gets more gradient, gets better, scores higher, and gets
even more tokens. Left alone, a router can end up sending almost everything to a few experts while the rest never
learn (stage 16 shows it). The model then has the cost of E experts and the quality of two.

### The Switch Transformer's loss

```
f_i = share of the N·k token slots that went to expert i      (counted: no gradient)
P_i = mean over tokens of the router's probability for i       (differentiable)
aux = E · Σ_i f_i · P_i
loss = language-model loss + aux_coef · aux                    (aux_coef = 0.01 here)
```

The gradient flows only through `P`. It lowers the probability of experts that are busy (big `f_i`) and so moves
tokens elsewhere. Two values are worth knowing by heart:

- **even routing** (`f_i = P_i = 1/E`): `aux = E · E · (1/E)(1/E) = 1`;
- **collapsed onto k experts** (`f_i = P_i = 1/k` for k of them): `aux = E · k · (1/k)² = E / k`.

So for 8 experts and top-2 the loss lives between about 1 and 4. (It can dip a little below 1 when the counts and
the probabilities disagree, which is why it is a nudge, not an exact measure.)

DeepSeek-V2 and OLMoE train with losses of this shape; DeepSeek-V3 keeps only a tiny sequence-level version and does
the main balancing with the bias of stage 09.

### Worked example

```bash
$ ./your_program.sh aux-loss
aux loss, even routing: 1.000
aux loss, collapsed onto 2 experts: 4.000 (E / k = 4.0)
random router with a head start for expert 0: aux loss 2.226
router gets a gradient: True
after 100 steps on the aux loss alone: 2.226 -> 0.983
expert 0's share of slots: 0.499 -> 0.173 (fair share 0.125)
```

A router with a head start for expert 0 puts it in almost every token's top-2 (share 0.499 of the slots, the most
top-2 allows). A hundred gradient steps on the aux loss alone bring that down to 0.173.

### The catch

The aux loss is added to the loss the model is trained on, so its gradient competes with the language-modelling
gradient. Too small a coefficient and it does not balance; too large and it balances by making routing worse for
the actual task. That tension is the reason for stage 09.

### Tests

The tests check 1.000 for even routing, `E / k` for collapsed routing (8 experts top-2 and 4 experts top-1), that the
router gets a gradient, and that 100 steps cut the loss below 1.2 and expert 0's share below 0.25.

### Check yourself

1. Why is `f_i` counted without a gradient? What is there to differentiate in a count?
2. Why multiply by E? (Hint: what would the even-routing value be without it, for E = 8 and for E = 64?)
