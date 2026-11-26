In this stage, you'll build the router: the part of an MoE layer that decides which experts each token visits.

### The host at a food court

Picture a food court with E stalls (the experts). A host reads each order (a token's vector) and sends it to the k
stalls it suits best. Every stall cooks only the orders it gets, so the food court can have many stalls while each
order costs only k stalls' work.

The router is one matrix, `E × d`:

```
scores  = W x                         (E numbers: how well each expert suits this token)
probs   = softmax(scores)             (sum to 1)
chosen  = the top_k largest probs     (which experts)
weights = chosen probs / their sum    (how much of each answer to keep; sum to 1)
```

Rescaling the chosen probabilities to sum to 1 is what DeepSeek-V3 and Mixtral do. With `top_k = 1` we keep the raw
probability instead, as the Switch Transformer does: a weight that is always 1.0 would give the router no gradient,
and it could never learn.

DeepSeek-V3 replaces the softmax by a sigmoid per expert (each expert scored on its own, not against the others);
DeepSeek-V2 and most other MoEs use the softmax. The rest of the course works with either.

### Worked example

Eight experts, top-2, six random tokens:

```bash
$ ./your_program.sh router
token 0: experts [2, 4] weights [0.88, 0.12]
token 1: experts [6, 1] weights [0.6, 0.4]
token 2: experts [4, 6] weights [0.51, 0.49]
token 3: experts [2, 1] weights [0.51, 0.49]
token 4: experts [2, 1] weights [0.79, 0.21]
token 5: experts [6, 0] weights [0.9, 0.1]
probabilities sum to 1: True
top-k weights sum to 1: True
experts per token distinct: True
chosen experts are the most probable: True
slots per expert: [1, 3, 3, 0, 2, 0, 3, 0]
```

The last line already shows the problem of stages 07-09: with 12 slots to fill, experts 3, 5 and 7 got nothing and
experts 1, 2 and 6 got three each.

### Tests

The tests run `router --tokens 10 --experts 6 --top-k 3`, parse every line, and check three distinct experts per
token, weights in decreasing order summing to 1, and 30 slots in all. A second test checks that a top-1 weight is a
real probability (between 0 and 1), not 1.0.

### Check yourself

1. Why must the router's choice (top-k) be made on the same scores the weights come from? What would happen if a
   token were sent to an expert whose weight was 0?
2. `topk` is not differentiable. How does the router learn at all? (Hint: follow the gradient through `weights`.)
