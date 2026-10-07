# Exercise 1 — Show coordinated omission (T0)

Run `examples/01_closed_vs_open.py` against the mock with `MOCKLLM_MAX_NUM_SEQS=8`, so it overloads easily. Then explain, in three sentences, why the closed-loop p99 is a lie.

**Test (`test_co.py`):** starts the mock with a small batch limit. A closed loop with 32 users saturates it, and an open-loop run then goes at the same mean rate. The test asserts that the open-loop p99 TTFT exceeds the closed-loop p99. Read the test: it is the shortest possible demonstration.
