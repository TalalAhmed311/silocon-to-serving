# Lane B / P5 local harness (planned)

This harness is built in Stage 4, at the first T2 module. The planned interface:

```cpp
// test side
auto r = s2s::check(solve, reference_cpu, inputs, {.rtol = 1e-5, .atol = 1e-6});
// bench side
s2s::bench("transpose", solve, inputs, {.warmup = 10, .reps = 100, .bytes = 2*N*N*4});
// prints: | size | median ms | p90 ms | GB/s | % of peak |  and writes results/<name>.json
```

- **Minimum toolchain:** CUDA 12.4, CMake 3.24.
- **Supported targets:** sm_75 (T4), sm_80, sm_86 (A10G), sm_89 (L4/L40S) and sm_90 (H100).
- **Peak values:** taken from `course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml`, where each entry cites its source, and from a measured copy kernel.
