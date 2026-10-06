# Exercise 2 — `histogram_quantile` like Prometheus (medium)

Prometheus stores a histogram as cumulative counts per upper bound `le`, ending with `+Inf`. Implement `histogram_quantile(q, buckets)` in `hq.py`, where `buckets` is a list of `(le, cumulative_count)` sorted by `le`, like this:

1. `rank = q × total`, where total is the `+Inf` count.
2. Find the first bucket whose cumulative count ≥ rank.
3. Interpolate linearly between that bucket's lower bound (the previous `le`, or 0 for the first) and its `le`.
4. If it is the `+Inf` bucket, return the largest finite `le`, as Prometheus does.
5. With total = 0, return `nan`.

Also implement `observe_into_buckets(values, les) -> list[(le, cum)]` to build test inputs.

**Test:** compared against `numpy.percentile` on 10⁴ lognormal latencies. The error must be within one bucket width. Also: an empty histogram, all samples in `+Inf`, and q = 0 / q = 1.

**Then answer:** with vLLM's default TTFT buckets, how wrong can a reported p99 be? (Look at the bucket list in `vllm/v1/metrics/loggers.py` at the pinned SHA.)
