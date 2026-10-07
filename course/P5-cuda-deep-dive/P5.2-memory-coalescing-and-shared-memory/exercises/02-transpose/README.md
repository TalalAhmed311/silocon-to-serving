# Exercise 2: transpose at ≥ 80% of copy bandwidth (T2, L2 exit check)

`kernel.cuh` is the naive transpose: coalesced reads, strided writes. Climb the ladder:

1. **Shared-memory tile:** load a 32×32 tile with coalesced reads, `__syncthreads()`, then write it out transposed with coalesced writes, reading the tile column-wise.
2. **Pad the tile** to `[32][33]`. The column-wise read was a 32-way bank conflict.

The test checks exact results on square, tall, wide and non-multiple-of-32 shapes. `--bench` at 8192² prints `exit check (>= 80% of copy): PASS` once you're there. Record each rung's GB/s, and confirm the bank-conflict change with `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`.
