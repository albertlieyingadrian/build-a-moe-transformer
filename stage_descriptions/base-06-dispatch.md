In this stage, you'll write the heart of the MoE layer: send every token to its chosen experts in batches
(**dispatch**) and add the weighted answers back in place (**combine**).

### Sorting the mail

The obvious version loops over tokens: for each token, for each of its k experts, run that expert on that one vector.
That is correct and painfully slow: an expert is a matrix multiply, and multiplying one vector at a time wastes it.

Instead, sort the mail by address. For each expert, gather every token that chose it into one batch, run the expert
once on that batch, and scatter the answers back to the tokens' rows:

```
for e in range(E):
    tok, slot = (idx == e).nonzero()           # which tokens chose e, and in which of their k slots
    out[tok] += w[tok, slot] * expert_e(x[tok])   # one batched call; index_add_ adds them back in place
```

A token that chose experts 2 and 5 is picked up in the loop for 2 and again in the loop for 5; its two answers land
on the same row and add up. On a GPU cluster the same idea becomes "all-to-all": every device sends each token to the
device that holds its expert, and the answers come back.

### Worked example

```bash
$ ./your_program.sh dispatch
tokens per expert: [16, 18, 17, 13]
total slots: 64 (tokens × top_k = 64)
max difference from the token-by-token reference: 1.5e-07
one expert, top-1: the layer is just that expert: True
```

32 tokens, 4 experts, top-2: 64 slots. The batched layer matches the slow loop to rounding error (`1e-7`). With a
single expert and top-1 the router's softmax over one expert is always 1.0, so the layer is exactly that expert.

### Tests

The tests run `dispatch --tokens 40 --experts 5 --top-k 2` and check 80 slots, a difference under `1e-5` from the
reference and the one-expert case.

### Check yourself

1. Why does the reference loop give the same numbers as the batched version, up to rounding?
2. If one expert gets 60 of 64 slots, how long does the layer take compared with even routing, if all experts run in
   parallel on different devices?
