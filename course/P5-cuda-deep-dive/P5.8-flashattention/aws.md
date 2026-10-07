# P5.8 on AWS

Follow [the shared P5 setup](../aws-common.md). The CUDA-core kernels and the exit check run on a **`g4dn.xlarge`** (T4). flash-attn and the tensor-core stretch need a **`g6.xlarge`** (L4). This module needs ≈ 3 instance-hours in total.

```bash
./build/p5/p5.8_attention_bench | tee results/p5.8-attention.md       # exit check: FA-2 vs naive at N = 4096
for x in 01-naive 02-fa1 03-fa2 04-paged-decode; do ./build/p5/p5.8_$x; done
sudo $(which ncu) -k regex:"attn_(scores|pv|fa2)" -c 3 --metrics dram__bytes.sum,gpu__time_duration.sum ./build/p5/p5.8_attention_bench
# L4 only:
uv venv && uv pip install torch && uv pip install flash-attn==2.8.3.post1 --no-build-isolation   # builds from source: slow
uv run python course/P5-cuda-deep-dive/P5.8-flashattention/examples/vs_flash_attn.py
```

Teardown and auto-stop: see the shared page.
