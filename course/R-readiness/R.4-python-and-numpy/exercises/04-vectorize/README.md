# Exercise 4 — Vectorise a loop (medium)

`distances.py` contains `pairwise_distances_loop`, a correct but slow double loop. Write `pairwise_distances` with the same result and no Python loops, by broadcasting `(n, 1, d)` against `(1, m, d)` and reducing the last axis. It must match the loop within `rtol=1e-6`.

Then measure it yourself. Do not assume a speed-up number: run it.

```python
import time, numpy as np
from distances import pairwise_distances, pairwise_distances_loop
a, b = np.random.rand(200, 8), np.random.rand(200, 8)
for f in (pairwise_distances_loop, pairwise_distances):
    t = time.perf_counter(); f(a, b); print(f.__name__, time.perf_counter() - t, "s")
```

Memory note: the broadcast version builds an `(n, m, d)` temporary. For `n = m = 2000, d = 2` in float64 that is 2000 x 2000 x 2 x 8 bytes = 64,000,000 bytes (about 61 MiB). Vectorising trades memory for speed.

**Test:** `uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/04-vectorize`
