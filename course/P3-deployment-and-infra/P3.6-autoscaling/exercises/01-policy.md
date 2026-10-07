# Exercise 1: the scaling policy as a pure function (T0)

`platform/autoscaler/policy.py` holds `raw_desired()` and `step()`. Read them, then make `test_policy.py` pass. It runs against **your** copy, `exercises/policy_impl.py`, which starts as a stub returning the current replica count. Set `S2S_SOLUTIONS=1` to run the tests against the reference instead.

The tests are the spec:

1. zero work → 0 replicas, after the cooldown only
2. `ceil(work / target)`, clamped to `[min, max]`
3. scale-up bounded by `max_up_per_step`
4. scale-down uses the **max** desired over the stabilization window and drops at most `max_down_per_step` per step
5. on a noisy trace, the stabilized policy changes the replica count less than half as often as the raw formula

**Then:** pick `target_per_replica` from your #4 knee (≈0.8 × knee concurrency) and write the number and its source into `scaledobject.yaml`'s comment.
