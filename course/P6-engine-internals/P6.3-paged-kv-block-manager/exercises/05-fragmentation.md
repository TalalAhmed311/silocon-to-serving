# Exercise 5 (hard): fragmentation and utilization under a trace (T0)

`bench/utilization.py` replays a synthetic trace (shared system prompt + varied user turns) through the engine with the fake model and reports slot utilization and prefix-hit rate per block size. Extend it to replay a **#4 loadgen trace** (P2.3's JSONL request log: prompt length, output length, arrival time — map arrival times to engine steps at a fixed step time).

Report, for block sizes 1, 8, 16, 32: mean and p99 slot utilization (used token slots ÷ allocated slots), prefix-hit rate, and preemptions. Then argue for a block size, and say what the GPU kernel side (P5.8) prefers and why the two pull in opposite directions.
