# P5.6 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4). Repeating on a `g6.xlarge` (L4) is a good comparison. This module needs ≈ 3 instance-hours.

```bash
NVIDIA_TF32_OVERRIDE=0 ./build/p5/p5.6_gemm_ladder | tee results/p5.6-ladder.md
uv run python course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/gemm_plot.py results/gemm.jsonl --out results/gemm.png
./build/p5/p5.6_01-gemm-rungs --bench && ./build/p5/p5.6_02-predication
for k in sgemm_r3 sgemm_r5 sgemm_r7; do sudo $(which ncu) -k regex:$k -c 1 --set full -o results/$k ./build/p5/p5.6_gemm_ladder 2048; done
uv run python course/P5-cuda-deep-dive/P5.6-gemm-ladder/examples/autotune.py --arch native     # ~15 min of builds
```

Teardown and auto-stop: see the shared page.
