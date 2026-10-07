# Exercise 2: pick a block size by occupancy, and justify it with numbers (T2)

1. Run `p5.1_02_occupancy`. For each of `light`, `heavy_regs` and `heavy_smem`, write down registers per thread, shared memory per block, and the occupancy at block sizes 64–1024.
2. Compile with `nvcc --resource-usage` (or read the `numRegs` the example prints) and explain the occupancy numbers by hand: `max blocks/SM = min(block limit, regs limit, smem limit, threads limit)`. Use the per-SM limits from `deviceQuery` or the CUDA Programming Guide table for your compute capability (cite which).
3. Time `heavy_regs` at block sizes 64, 128, 256 and 512 (copy the `time_gpu` pattern from `03_divergence.cu`). Is the fastest block size the highest-occupancy one? Explain.
4. Add `__launch_bounds__(256, 4)` to `heavy_regs`. What happened to registers per thread and to time (spills? check `-Xptxas -v`)?

Deliverable: a 6–10 line justification of the block size you'd ship for each kernel.
