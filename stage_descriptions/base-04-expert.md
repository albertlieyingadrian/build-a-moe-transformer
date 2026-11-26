In this stage, you'll build one expert: a SwiGLU feed-forward network. An MoE layer is just many of these and a
router.

### A dimmer switch on every wire

A classic transformer FFN is `down(gelu(up(x)))`: widen, bend, narrow. SwiGLU adds a second widening, the **gate**,
and multiplies:

```
expert(x) = down( silu(gate(x)) * up(x) )        silu(z) = z · sigmoid(z)

     x (d) ──► gate ──► silu ──┐
       │                       × ──► down ──► (d)
       └────► up   ────────────┘
```

`up(x)` carries the content; `silu(gate(x))` decides, number by number, how much of it to let through. Think of the
up-projection as h wires and the gate as a dimmer switch on each one. A closed gate (`gate(x) = 0`, so `silu = 0`)
lets nothing through. SwiGLU is what DeepSeek, Llama and Mixtral use for every expert.

### Counting weights

Three matrices, `d × h` each, no biases: **`3 · d · h` weights.** A two-matrix GELU MLP has `2 · d · h'`, so the
GELU network with the same weights is 1.5 times wider. Keeping that count straight matters: from stage 10 on, every
comparison is "at the same number of weights".

### Worked example

```bash
$ ./your_program.sh expert
expert parameters: 24576 (3 × d × h = 24576)
a GELU MLP (two matrices) with the same parameters has hidden width: 192
output shape: (5, 64)
silu(-3), silu(0), silu(3): -0.142 0.000 2.858
output with the gate closed (gate weights 0): 0.0
```

### Tests

The tests run `expert --d 48 --h 96` and check the count `3 · 48 · 96`, the GELU width 144, the shape, the three
`silu` values and the zero output with the gate closed.

### Check yourself

1. `silu(-3)` is slightly negative. What does that let the gate do that a hard on/off switch (ReLU) can't?
2. If you replace one expert of width h by two experts of width h/2, how many weights do you have?
