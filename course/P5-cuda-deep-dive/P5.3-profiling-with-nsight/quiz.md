# P5.3 quiz

<details><summary><b>1.</b> nsys or ncu: "the GPU is idle 40% of each decode step"?</summary>

nsys: it's a timeline question (CPU gaps, launches, syncs), not a kernel question.
</details>

<details><summary><b>2.</b> Speed-of-Light shows memory 15%, compute 10%. What next?</summary>

Latency-bound. Check achieved occupancy and the top warp-stall reasons, and check whether the grid is too small for the GPU.
</details>

<details><summary><b>3.</b> Top stall reason "Long Scoreboard". Meaning?</summary>

Warps are waiting on global (or local) memory loads. Improve coalescing and reuse (shared memory), or add parallelism (occupancy, ILP) to hide latency.
</details>

<details><summary><b>4.</b> Why don't ncu durations match your CUDA-event timings?</summary>

ncu serializes and replays kernels, flushes caches between passes by default, and adds overhead. Use it for metrics. Use events or nsys for timing.
</details>

<details><summary><b>5.</b> What does `ERR_NVGPUCTRPERM` mean, and the two fixes?</summary>

The process isn't allowed to read GPU performance counters. Run ncu with sudo, or set the nvidia module option `NVreg_RestrictProfilingToAdminUsers=0` and reload the driver (single-user boxes only).
</details>
