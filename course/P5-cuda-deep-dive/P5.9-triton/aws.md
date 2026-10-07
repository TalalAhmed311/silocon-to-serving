# P5.9 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g6.xlarge`** (L4: `tl.dot` on tensor cores with bf16/fp16). This module needs ≈ 2 instance-hours.

```bash
uv sync --extra torch                      # torch brings its matching triton; check: python -c "import triton; print(triton.__version__)"
uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch
PYTHONPATH=platform uv run --extra torch python course/P5-cuda-deep-dive/P5.9-triton/examples/bench_triton_vs_cuda.py | tee results/p5.9.md
```

If the installed Triton isn't `v3.8.0`, note the version next to your numbers. API details (`input_precision`, autotune keys) can differ between versions.

Teardown and auto-stop: see the shared page.
