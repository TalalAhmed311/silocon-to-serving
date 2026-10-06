# Exercise 1 — PromQL p90 TTFT and the SLI, tested (T0)

1. Install `promtool`, which ships in the Prometheus release tarball, and run `promtool test rules platform/observability/rules/slo_test.yaml`.
2. Add a test case for the recording rule `s2s:ttft_seconds:p90_5m`. Feed three buckets (`le="0.5"`, `le="2"`, `le="+Inf"`) whose rates put 50% of requests under 0.5 s and 95% under 2 s, then assert the p90 value `histogram_quantile` computes. Work it out by hand first, using P1.3 exercise 2's interpolation rule.
3. `test_promtool.py` runs promtool from pytest, and skips if promtool isn't on your PATH.
