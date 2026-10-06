# Exercise 3 — #4 on the mock (medium)

`platform/loadgen` is #4. Read `workload.py`, `runner.py` and `analysis.py` first: they are short. Then:

1. Run the walkthrough sweep against `mockllm` and paste the table into your notes.
2. **Test:** `test_loadgen_mock.py` starts the mock with a fast `TIME_SCALE`, sweeps 6 rates, and checks that the knee `find_knee` reports is within ±25% of the closed-form saturation rate `max_num_seqs / (step_ms × mean_output_tokens)`. Read the test to see the exact numbers.
3. Extend loadgen with **one** feature, with a test:
   - a `--ramp` mode that increases the rate continuously instead of in steps
   - a per-request CSV dump
   - a "think time" closed-loop mode, clearly labelled, for comparison in P2.4
