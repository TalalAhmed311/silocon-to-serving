# 81: INT4 Weight-Only Quantized MatMul (L5, stretch: hints and outline only)

Problem: [LeetGPU #81](../../leetgpu-map.md). `y = x · dequant(W_int4)` with fp16/fp32 activations and int4 weights + group scales (W4A16).

**Hint ladder**

1. At decode, `x` is 1–64 rows, so this is a **GEMV/skinny GEMM, memory-bound on the weights**. int4 moves 1/4 of fp16's bytes, and that's the whole speedup.
2. **Never write the dequantized W.** Load packed bytes, unpack two nibbles per byte in registers (`(b & 0xF) − 8`, `(b >> 4) − 8` for symmetric int4, or subtract a zero-point), multiply by the group scale, and FMA into the accumulator (L4 #64 + #22).
3. Layout matters. Put consecutive k for one output column in one 32-bit word (8 nibbles), so one load feeds 8 FMAs. Production formats (GPTQ/AWQ/Marlin) also **interleave** for tensor-core fragment layouts.
4. For tensor cores, dequantize into fp16 fragments right before `mma.sync` (Marlin-style). That's an advanced exercise: read the Marlin paper/repo.

**Solution outline:** one warp per output column (or 4 columns), lanes striding over K in 8-nibble words, the group scale fetched per group, shuffle-reduce at the end. Bench it against fp16 GEMV at batch 1: you should see close to 4× on the weight-read time.
