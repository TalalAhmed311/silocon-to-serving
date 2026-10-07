# Exercise 2: NVTX-annotate the inference loop and read the nsys timeline (T2)

1. Run `examples/02_nvtx_ranges.cu` under `nsys profile` and find the "prefill" and "decode step" ranges, the kernels inside them, and the CPU gaps between decode steps.
2. Add NVTX ranges to the #0 engine (`platform/engine/v0/src/engine.hpp`): `prefill`, `decode step`, and per layer `attention` / `mlp`. Include `<nvtx3/nvToolsExt.h>` from the CUDA toolkit. It's header-only, and Nsight Systems also traces CPU-only programs. Guard it with `#ifdef S2S_NVTX`, so the T0 CPU build doesn't need CUDA.
3. Profile one 32-token generation: `nsys profile -o engine ./build/engine/s2s-engine …`. Which phase dominates? Does the time per decode step grow with position (attention over the KV cache)?

Deliverable: a timeline screenshot with the ranges visible, plus 3–5 sentences on what it shows. (#0 v1 in P6 is the GPU version. Keep these ranges, because P6.6 uses the same view to show what CUDA graphs remove.)
