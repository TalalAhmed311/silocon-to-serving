# Exercise 4 — Measure achieved bandwidth (hard, T2)

Write `measure_copy_gbs(n_bytes) -> float` in `bw.py` with PyTorch on CUDA:

- allocate two buffers and warm up
- time 20 copies with CUDA events and take the median
- count read + write bytes

Then compare it with `gpu_specs.yaml` and with `examples/03_measure_bw.py`. If you've done Lane B L1, compare with the harness's `s2s::measure_copy_gbs()` too, a hand-written `float4` copy kernel.

**Test:** skipped without CUDA. On a GPU it requires ≥ 60% of the spec bandwidth for a 1 GiB copy. If the GPU isn't in the YAML, it only requires > 0.

`TODO(run-on: g6.xlarge)`: run `uv run --extra torch pytest course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/exercises/04-measure-bw` and record the GB/s in `GAPS.md` §C.
