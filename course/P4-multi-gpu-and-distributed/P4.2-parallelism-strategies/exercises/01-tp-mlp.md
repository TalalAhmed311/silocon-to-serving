# Exercise 1: TP MLP numerics match unsplit (T0)

Implement `tp_mlp(x, w_gate, w_up, w_down, tp)` in `tp_mlp.py`. Return the list of per-rank **partial** outputs and their sum (the all-reduce).

`test_tp_mlp.py` checks, for tp ∈ {1, 2, 4, 8} and 1 or 17 tokens:

- the sum matches the unsplit MLP at fp32 with `atol = rtol = 1e-5`
- every partial has the full output shape, and a single partial is **not** the answer (the all-reduce is required)
- the wrong split order (row first) gives a different answer

**Then:** extend it to attention. Split heads across ranks (W_q, W_k, W_v by columns grouped by head, W_o by rows), run causal attention per rank on its heads, and show that the sum of partials equals the unsplit attention output. What happens with GQA when `kv_heads < tp`?
