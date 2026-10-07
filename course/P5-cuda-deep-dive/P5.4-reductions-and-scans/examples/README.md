# P5.4 examples (T2, sm_75+)

| File | Shows |
|---|---|
| [`01_reduce_ladder.cu`](01_reduce_ladder.cu) | Harris's ladder (6 rungs from `d4/reduce.cuh`) vs `cub::DeviceReduce::Sum`, in GB/s and % of read bandwidth |
| [`02_warp_reduce.cu`](02_warp_reduce.cu) | one warp reduction three ways: smem + `__syncwarp`, `__shfl_down_sync`, `cg::reduce` |
| [`03_scan.cu`](03_scan.cu) | 3-phase scan vs single-pass decoupled look-back vs `cub::DeviceScan` |

CUB comes with the CUDA toolkit (CCCL). The course pins CCCL `v3.5.0` for reading. The toolkit's bundled version is the one you compile against: print `CUB_VERSION` and note it next to your numbers.
