# Exercise 2 — When does decode become compute-bound? (medium)

Implement `crossover_batch(P, d, L, kv_bytes_per_token, t, peak_flops, bw, bytes_per_param=2, max_batch=4096)`. It returns the smallest batch B whose decode intensity reaches the GPU's ridge `peak_flops / bw`, or `None` if no B ≤ `max_batch` does. Use the model from §3 of the lesson:

```
FLOPs/step = 2·P·B + B·4·d·L·t        bytes/step = P·bytes_per_param + B·t·kv_bytes_per_token
```

**Test:** fixture GPUs with round, made-up numbers (the test doesn't depend on real specs). It checks short context (crosses at a predictable B), long context (never crosses), and monotonicity in t.

**Then explain** in two sentences why the long-context case can never cross, whatever the batch. Hint: take B → ∞ in the intensity formula.
