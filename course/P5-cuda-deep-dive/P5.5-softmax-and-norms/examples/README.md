# P5.5 examples (T2, sm_75+)

| File | Shows |
|---|---|
| [`01_softmax_ladder.cu`](01_softmax_ladder.cu) | safe 3-pass vs online 2-pass vs row-in-registers softmax, in GB/s, for short, medium and very long rows |
| [`02_rmsnorm.cu`](02_rmsnorm.cu) | RMSNorm fp32 scalar vs `float4` vs bf16 storage, against copy bandwidth |
| [`03_fused_residual_rmsnorm.cu`](03_fused_residual_rmsnorm.cu) | residual add + RMSNorm as 2 kernels vs 1 fused kernel |
| [`04_fp16_bf16.cu`](04_fp16_bf16.cu) | rounding error and range of fp16 vs bf16: where test tolerances come from |

Kernels: `platform/kernels/include/d4/softmax.cuh`, `norms.cuh`.
