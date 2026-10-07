# P5.2 examples (T2, sm_75+)

| File | Shows |
|---|---|
| [`01_strided_read.cu`](01_strided_read.cu) | useful GB/s vs stride and misalignment, next to the sectors-per-request you should predict |
| [`02_transpose_ladder.cu`](02_transpose_ladder.cu) | the three transpose rungs from `d4/transpose.cuh` vs measured copy bandwidth |
| [`03_bank_conflicts.cu`](03_bank_conflicts.cu) | row vs column shared-memory reads, with and without +1 padding |
