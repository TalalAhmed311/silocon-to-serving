# Exercise 1: rungs 3–7 from starters, each tested against cuBLAS (T2, L5 exit check)

`kernel.cuh` has rungs 1–2. Implement 3–7 behind `sgemm(rung, …)`, with the size contract in `sgemm_supported`. Each rung has its own test (`rung3` … `rung7`) against cuBLAS SGEMM with `rtol = atol = 1e-3`: fp32 with a different summation order, on K up to 256.

For **every** rung, before running it:

1. **Predict** the speedup over the previous rung from what changes: global loads per FMA, shared-memory loads per FMA, coalescing.
2. **Measure** with `--bench` (TFLOP/s and % cuBLAS at N = 1024, 2048, 4096).
3. **Explain** the change with **one** ncu metric, for example `dram__bytes.sum` (rung 2→3), `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` (3→4→5), `smsp__inst_executed_op_ld.sum` (5→6), or `smsp__warp_issue_stalled_*` (6→7).

`--bench` prints `exit check (best >= 70% of cuBLAS): PASS` when the L5 SGEMM target is met. `TODO(run-on: g4dn.xlarge)`. Then run `examples/autotune.py` and see whether a different rung-7 configuration wins on your GPU.
