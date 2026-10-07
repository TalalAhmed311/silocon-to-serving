# P5.1 quiz

<details><summary><b>1.</b> blockDim = (16, 8). Which warp is thread (5, 3) in?</summary>

Linear id = 3·16 + 5 = 53, so warp 1 (lane 21).
</details>

<details><summary><b>2.</b> Why use a grid-stride loop instead of one thread per element?</summary>

It works for any n with a grid sized to the GPU, amortizes per-thread setup, and avoids grid-size limits and int overflow.
</details>

<details><summary><b>3.</b> When does higher occupancy *not* help?</summary>

When latency is already hidden by ILP or enough warps, or when the kernel is limited by something else (bandwidth, compute). Register-blocked GEMMs trade occupancy for reuse on purpose.
</details>

<details><summary><b>4.</b> Kernel faults show up "in the wrong place". Why?</summary>

Launches are asynchronous. The error is reported by the next synchronizing API call. Use `cudaGetLastError` + synchronize in tests, or `CUDA_LAUNCH_BLOCKING=1` when debugging.
</details>

<details><summary><b>5.</b> What two conditions does copy/compute overlap need?</summary>

`cudaMemcpyAsync` from pinned host memory, with the copy and the kernel in different (non-default, or properly ordered) streams.
</details>
