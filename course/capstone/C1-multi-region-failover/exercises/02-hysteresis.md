# Exercise 2: a health check that doesn't flap (T0)

`HealthTracker` has three knobs: `fail_threshold`, `rise_threshold`, `latency_slo_s`. With probes every 10 s:

1. Compute the detection time and the expected false-down rate per day under 5 % independent probe loss for fail_threshold ∈ {1, 2, 3, 5} (P(k consecutive failures) per probe ≈ (1 − p)·p^k). Simulate it to check your formula (adapt `test_flapping_bounded_under_noise`).
2. Pick values that detect a real outage within 60 s and false-alarm less than once a month. Write them into `prober.py`'s defaults and `infra/aws/failover` (`alarm_periods`) consistently — the alarm adds its own hysteresis on top.
3. Add a test that encodes your requirement.
