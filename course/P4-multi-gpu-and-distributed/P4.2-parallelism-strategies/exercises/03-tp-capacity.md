# Exercise 3: TP communication in the D3 capacity calculator (T0)

D3 (`platform/capacity/core.py`) used to assume TP was free. It now accepts `link_gbs` and `link_alpha_us`, and adds `tp_comm_seconds_per_step` to each decode step: 2 all-reduces per layer of `tokens × hidden × 2 B`, costed with P4.1's ring model.

1. Implement the function yourself in `tp_comm.py`. `test_tp_capacity.py` compares it with the reference, checks that batch-1 decode is latency-dominated, and checks that `plan()` slows down once comm is included.
2. Use your fitted α and B from P4.1 (PCIe box and, if you ran it, NVLink box) and print the batch-1 and batch-32 decode ceilings for TP = 1, 2, 4. Paste the table into the README bench.
3. **Extend:** add prefill. During prefill, the all-reduce size is `prompt_tokens × hidden × 2 B`. Is it bandwidth-bound now? Add the term to `prefill_s` and add a test.
