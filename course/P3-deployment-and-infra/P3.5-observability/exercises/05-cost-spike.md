# Exercise 5 — Cost-spike alert on a synthetic tenant flood (T0)

Write `platform/observability/rules/cost_test.yaml`. Give tenant `a` a steady 1,000 tokens/min for 25 hours, then 10,000 tokens/min for 30 minutes. Give tenant `b` a steady 1,000 tokens/min throughout. Assert that `TenantTokenSpike` fires for `a` only, after its `for: 15m`. `test_promtool.py` picks up every `*_test.yaml` automatically.

Then answer: why does the rule compare against `offset 1h` of a 1-day rate, instead of the last day including the spike?
