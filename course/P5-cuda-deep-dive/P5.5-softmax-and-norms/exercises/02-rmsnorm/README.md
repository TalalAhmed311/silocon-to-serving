# Exercise 2: RMSNorm within X% of copy bandwidth (T2)

Implement `rmsnorm` in `kernel.cuh`: `y = x · rsqrt(mean(x²) + eps) · w`, fp32, one block per row, `float4` loads when `cols % 4 == 0`, fp32 accumulation.

**Setting X.** RMSNorm reads the row (and `w`, which stays in cache) and writes it once, so its floor is the copy kernel. First measure your correct-but-simple version with `--bench`, then the copy bandwidth (the harness prints it in the "% of peak" column). Set **X = the gap you commit to closing**, e.g. "within 15% of copy at hidden 4096". Write X and the measured baseline in your notes **before** optimizing. Then get there: vectorize, pick the block size by row length, keep the row in registers so it's read once.

The test is fp32 with `rtol = 1e-5`. The bf16/fp16 tolerances are derived in exercise 3.
