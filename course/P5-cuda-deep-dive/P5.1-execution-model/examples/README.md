# P5.1 examples (T2, sm_75+)

Build them all with `course/P5-cuda-deep-dive/CMakeLists.txt`. The binaries are `build/p5/p5.1_<name>`.

| File | Shows |
|---|---|
| [`01_hello_indices.cu`](01_hello_indices.cu) | block/thread → global index → warp and lane, plus which SM ran each block (`%smid`) |
| [`02_occupancy.cu`](02_occupancy.cu) | the occupancy API on three kernels limited by nothing, by registers, and by shared memory |
| [`03_divergence.cu`](03_divergence.cu) | lane-divergent vs warp-uniform branches vs predicated small branches |
| [`04_pinned_vs_pageable.cu`](04_pinned_vs_pageable.cu) | H2D/D2H GB/s from pageable vs pinned memory (P0.2 revisited) |
| [`05_streams_overlap.cu`](05_streams_overlap.cu) | copy/compute overlap with 3 streams, made to be read in Nsight Systems |
