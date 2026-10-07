# P5.6 examples (T2, sm_75+)

| File | What |
|---|---|
| [`gemm_ladder.cu`](gemm_ladder.cu) | one runner for rungs 1–7 of `d4/gemm.cuh` plus cuBLAS: `N | rung | ms | TFLOP/s | % cuBLAS | % peak` |
| [`gemm_plot.py`](gemm_plot.py) | TFLOP/s vs N per rung from `results/gemm.jsonl` |
| [`autotune.py`](autotune.py) | builds and times every valid rung-7 tile configuration on your GPU and keeps the best |

The rung kernels themselves live in [`platform/kernels/include/d4/gemm.cuh`](../../../../platform/kernels/include/d4/gemm.cuh), one function per rung, each commented with what it changes. Read them alongside siboehm's article: same order, original code.
