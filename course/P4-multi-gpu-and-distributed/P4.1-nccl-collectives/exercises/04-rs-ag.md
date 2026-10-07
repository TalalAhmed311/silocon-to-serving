# Exercise 4 (hard): reduce-scatter + all-gather = all-reduce

1. **Simulator.** `test_rs_then_ag_equals_all_reduce` checks the identity on 6 ranks: same result, same step count. Make it pass with your `ring.py`.
2. **Timing (T3).** From your nccl-tests runs: is `t(all_reduce, S) ≈ t(reduce_scatter, S) + t(all_gather, S)` across sizes? Make a table. Where does it break down (small sizes: two launches pay α twice), and by how much?
3. **Why it matters.** FSDP/ZeRO replaces DP's all-reduce of gradients with a reduce-scatter (each rank keeps its gradient shard) plus an all-gather of parameters before each forward. It's the **same bytes**, split in time, and that's why sharding is "free" in communication volume (P4.2).
