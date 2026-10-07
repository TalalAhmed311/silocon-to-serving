# Exercise 3: remove the bank conflicts from a planted kernel (T2)

`kernel.cuh` computes per-tile row sums through shared memory. The global load is coalesced. The shared-memory read `t[tx][k]` puts every lane in the same bank (word `32·tx + k` → bank `k`).

1. Measure the conflict count: `sudo ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/p5/p5.2_03-bank-conflicts --bench`.
2. Fix it without changing results (the test is exact). Padding is one fix. Can you also fix it by changing *which* thread reads what, with no padding?
3. Re-measure the counter (should be ~0) and the time.
