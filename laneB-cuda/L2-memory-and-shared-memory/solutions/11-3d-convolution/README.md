# 11 — 3D Convolution (L2, stretch)

Problem: [LeetGPU #11](../../leetgpu-map.md). Valid 3-D correlation of a `D×H×W` volume with a `KD×KH×KW` kernel.

**Hint ladder**

1. Extend #10 by one dimension: one thread per output voxel. With a 3-D grid you get z from `blockIdx.z`.
2. A full 3-D shared tile with halo gets large fast: 16×16×16 + halo is > 16 KB. Instead, tile in x–y and **stream along z**: keep a rolling window of KD planes in registers or shared memory.
3. The kernel goes in `__constant__`, as before.

**Solution outline (this file):** the direct version with a 3-D grid and constant-memory weights. Implementing the streaming-along-z optimization is the stretch goal: compare it with `--bench`.
