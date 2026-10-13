# Exercise 3: a two-layer MLP

Implement `silu`, `mlp_forward` and `param_count` in `mlp.py`. The network is `y = W2 @ silu(W1 @ x + b1) + b2`. Parameters are the numbers inside `W1`, `b1`, `W2` and `b2`; the test checks your count against hand arithmetic (for 3 -> 4 -> 2: 12 + 4 + 8 + 2 = 26).

```bash
uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/03-mlp-forward
```
