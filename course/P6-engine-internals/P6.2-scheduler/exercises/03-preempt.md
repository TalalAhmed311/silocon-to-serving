# Exercise 3: preemption by recompute (T0)

When a **running** sequence can't get blocks for its next tokens, preempt: remove the victim from `running`, `blocks.free_seq(victim)`, set `num_computed = 0`, `num_preemptions += 1`, put it back in `waiting`. Victim = lowest priority, newest admitted, **never** a sequence already scheduled in this step; if none, the sequence preempts itself. Skip admissions in a step that preempted.

Tests: `test_preempt_under_memory_pressure` (8 blocks of 4 tokens, outputs must still equal the reference) and `test_preempt_happens`.

**Then:** write the liveness argument in three sentences. Which property of `LLMEngine.add_request` does it rely on? (Read `engine.py`.) What breaks if a waiting sequence keeps its prefix-hit blocks after a failed admission?
