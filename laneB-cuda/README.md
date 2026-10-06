# Lane B — CUDA C++ on LeetGPU

Lane B runs in parallel with Lane A for all 44 weeks. It does 3–4 problems a week during P0–P4 (weeks 1–24) and daily from P5 (week 25) on. Write every kernel in **CUDA C++ first**. Triton versions come in L6 and in project #12.

| Level | Weeks | Folder | Exit check |
|---|---|---|---|
| L1 | 1–6 | [L1-launch-and-indexing](L1-launch-and-indexing/syllabus.md) | write 2D indexing for any shape unaided |
| L2 | 7–14 | [L2-memory-and-shared-memory](L2-memory-and-shared-memory/syllabus.md) | transpose ≥80% of copy bandwidth |
| L3 | 15–24 | [L3-reductions-and-scans](L3-reductions-and-scans/syllabus.md) | reduction within 10% of `cub::DeviceReduce` |
| L4 | 25–27 | [L4-fused-elementwise-and-norms](L4-fused-elementwise-and-norms/syllabus.md) | single-pass online softmax, reported in GB/s |
| L5 | 28–31 | [L5-gemm-and-tensor-cores](L5-gemm-and-tensor-cores/syllabus.md) | SGEMM ≥70% cuBLAS; tensor-core HGEMM ≥50% |
| L6 | 32–44 | [L6-attention-and-triton](L6-attention-and-triton/syllabus.md) | FA-2 forward ≥5× naive at seq 4k |

Optional problems that aren't on the inference path are in [electives.md](electives.md).

## How each level folder is organised

- `leetgpu-map.md` lists the real challenges: title, link, difficulty, concept and priority.
- `solutions/<id>-<slug>/` holds, per problem:
  - `README.md`: an **original** 3-step hint ladder, a solution outline and a "why this is fast" note
  - `kernel.cu`: your solution
  - `test.cpp`: a local harness test
- `bench/` holds one script per level that runs every solved kernel through the harness and writes a Markdown table and JSON. `ncu --csv` summaries from real runs are committed beside the results.

## Two ways to run a kernel

1. **T1, in the browser.** Paste the kernel into the LeetGPU playground.
2. **T2, on a local GPU or AWS.** Run it through `harness/`:
   - A CMake + CTest project that calls your `solve(...)` with the same signature style.
   - It checks the result against a CPU, cuBLAS or CUB reference with stated `rtol`/`atol`.
   - It benchmarks with warm-up, repetitions and median/p90.

The harness inputs are our own. LeetGPU's tests are not copied, because the challenge repo is CC BY-NC-ND 4.0.

## Hardware labels

- L1–L4 run fine on T1 (LeetGPU) or a T4 (`g4dn.xlarge`, sm_75).
- L5's tensor-core rungs need sm_80+ for `mma.sync` bf16 and `cp.async`: an A10G (`g5`) or L4 (`g6`). An L4 is needed for FP8.
