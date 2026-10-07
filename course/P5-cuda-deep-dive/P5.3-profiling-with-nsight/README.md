# P5.3: Profiling with Nsight Compute and Nsight Systems

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) (needs GPU counter access. See [the shared P5 setup](../aws-common.md#nsight-compute-counters-err_nvgpuctrperm)). The CSV tooling is T0 |
| **Time** | ≈25 min reading + ≈8 h hands-on |
| **Prerequisites** | P5.1, P5.2 |
| **Reading** | Nsight Systems and Nsight Compute user guides (Profiling Guide: metrics and sections) · GPU MODE profiling lecture |

## Learning objectives

1. Use **Nsight Systems** for the whole-program view: timelines, CPU/GPU gaps, streams, NVTX ranges.
2. Use **Nsight Compute** for one kernel: Speed-of-Light, memory workload, occupancy, warp-stall reasons, source-level hotspots, roofline.
3. Get profiling working on a cloud box (`ERR_NVGPUCTRPERM`), and move reports to a laptop UI.
4. Turn ncu CSV output into the "% of peak" column that every D4 bench table carries.

---

## 1. Which tool, when

| Question | Tool |
|---|---|
| Where does wall-clock time go? Is the GPU idle between kernels? Do copies overlap? | **nsys** (`nsys profile -o out ./bin`) |
| Why is *this kernel* slow? Memory- or compute-bound? What do warps stall on? | **ncu** (`ncu --set full -o out ./bin`, or `-k regex:name`) |

Start with nsys. Optimizing a kernel that's 3% of the timeline is wasted effort, and decode loops are often bound by launch overhead and CPU gaps, not kernels (`02_nvtx_ranges.cu`, then CUDA graphs in P6.6).

## 2. Reading an ncu report, top-down

1. **GPU Speed of Light:** memory % and compute % of peak. Both low means latency-bound: occupancy, stalls, or too little work.
2. **Memory Workload Analysis:** sectors per request (P5.2 coalescing), L1/L2 hit rates, bank conflicts.
3. **Occupancy:** theoretical vs achieved, and the limiter (registers, shared memory, block size).
4. **Warp State Statistics:** the top stall reason. *Long Scoreboard* means waiting on global memory. *MIO Throttle* or *Short Scoreboard* means shared memory or special functions. *LG Throttle* means too many memory instructions. *Barrier* means `__syncthreads` imbalance.
5. **Source** view: the SASS/CUDA line with the most stall samples. Compile with `-lineinfo` (the P5 CMake does).
6. **Roofline** chart: where the kernel sits relative to the P1.4 ridge point.

## 3. On a cloud instance

Counters are admin-only by default: `sudo $(which ncu) …`, or the `NVreg_RestrictProfilingToAdminUsers=0` module option plus a reboot (see the shared setup). Reports are binary and large: write them with `-o`, copy them back, open them in the free desktop UIs. Commit CSV summaries, never reports.

> **Predict first.** `mystery_a` walks one row per thread. Rows are 256 floats (1 KB) apart. Predict sectors per request, and the fraction of the reference kernel's bandwidth you expect, before opening its report.

---

## Walkthrough

```bash
sudo $(which ncu) --set full -o results/planted ./build/p5/p5.3_01_planted_slow
nsys profile -o results/nvtx ./build/p5/p5.3_02_nvtx_ranges
uv run pytest course/P5-cuda-deep-dive/P5.3-profiling-with-nsight/exercises
```

## What you should see

- `01`: the reference near copy bandwidth, and each mystery kernel several times slower, for three different reasons visible in three different ncu sections.
- `02`: decode-step ranges much wider than the sum of their kernels. The gaps are launch overhead plus `cudaDeviceSynchronize`.

`TODO(run-on: g4dn.xlarge)`

## Exercises

| # | Exercise | Test |
|---|---|---|
| 1 | [Diagnose 3 planted slow kernels from ncu alone](exercises/01-diagnose.md) | written table + fixes |
| 2 | [NVTX-annotate the #0 inference loop and read the nsys timeline](exercises/02-nvtx-engine.md) | screenshot + notes |
| 3 | [`ncu --csv` → "% of peak" column](exercises/03-ncu-csv.md) | `test_ncu_to_table.py` |

## Common mistakes

- Profiling a debug build (`-G`): every number is wrong. Use Release + `-lineinfo`.
- Reading ncu kernel times as end-to-end: ncu replays each kernel many times with caches flushed. Use nsys or CUDA events for timing.
- Optimizing the top stall reason without checking Speed-of-Light first.
- Leaving `NVreg_RestrictProfilingToAdminUsers=0` on a shared machine.

## Go deeper

- Nsight Compute Profiling Guide (metrics, sections, roofline) and Nsight Systems user guide.
- GPU MODE lecture on profiling (path in `gpu-mode/lectures`, verified at build time).

**Next:** [P5.4 Reductions and scans](../P5.4-reductions-and-scans/README.md).
