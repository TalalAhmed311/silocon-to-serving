# Lane B / P5 local harness

This is the local counterpart of the LeetGPU judge. Every solution in `L*/solutions/<id>-<slug>/` has:

| File | What |
|---|---|
| `README.md` | an original hint ladder (3 rungs), a solution outline, and a "why this is fast" note |
| `kernel.cu` | the solution: `solve(...)` launches the kernel(s) on **device** pointers, matching how LeetGPU calls you |
| `test.cu` | builds random inputs, runs `solve`, and compares against a CPU (or cuBLAS/CUB) reference with stated tolerances. `--bench` prints a timing row |

## Build and run (T2: needs an NVIDIA GPU, CUDA ≥ 12.4)

```bash
cmake -S laneB-cuda -B build/laneB -DCMAKE_BUILD_TYPE=Release        # CMAKE_CUDA_ARCHITECTURES defaults to "native"
cmake --build build/laneB -j
ctest --test-dir build/laneB --output-on-failure                       # correctness for every solved problem
./build/laneB/L2_3-matrix-transpose --bench                           # timing table + results/<bench>.jsonl
```

On AWS, use `infra/aws/single-node` with `instance_type = "g4dn.xlarge"` (T4, sm_75) for L1–L4, and `g6.xlarge` (L4, sm_89) for L5 tensor cores and FP8. Remember `make down`.

## The API (`harness/include/s2s_cuda.cuh`)

```cpp
CUDA_CHECK(cudaMalloc(...));               // aborts with file:line and the CUDA error name
CUDA_CHECK_LAUNCH();                       // after a <<<>>> launch in tests: surfaces async faults
s2s::DeviceBuffer<float> d(host_vec);      // RAII cudaMalloc + upload; d.download() -> std::vector
auto t = s2s::time_gpu([&]{ solve(...); });        // CUDA events, 5 warm-up, 50 reps -> median/p90 ms
double copy = s2s::measure_copy_gbs();              // measured "100%" for memory-bound kernels
s2s::report("transpose", "smem+pad", N, t, gbs, "GB/s", copy);   // Markdown row + results/transpose.jsonl
```

**Tolerances.** fp32 elementwise ops: `rtol = 1e-5`. Reductions and dot products: `rtol = 1e-4`, `atol` scaled by √N, because the GPU sums in a different order from the CPU reference. fp16: `rtol = 1e-2` (10-bit mantissa). Each test states its own tolerance and why.

**Copyright.** The inputs, references and tests here are ours. LeetGPU's statements, starters and tests are CC BY-NC-ND 4.0 and are not reproduced. Open each problem on leetgpu.com from the level's `leetgpu-map.md`.
