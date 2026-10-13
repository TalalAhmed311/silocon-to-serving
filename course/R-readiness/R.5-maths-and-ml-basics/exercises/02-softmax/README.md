# Exercise 2: stable softmax

Implement `softmax(x, axis=-1)` in `softmax.py`. Softmax is `exp(x) / sum(exp(x))`, but `exp(1000)` overflows in float32 (the largest float32 is about 3.4e38, and `exp(89)` already exceeds it). Subtracting the maximum first gives the same answer, because the shift cancels in the ratio, and keeps every exponent at 0 or below. The test uses logits like `[1000, 1001]` and also checks several axes.

```bash
uv run pytest course/R-readiness/R.5-maths-and-ml-basics/exercises/02-softmax
```
