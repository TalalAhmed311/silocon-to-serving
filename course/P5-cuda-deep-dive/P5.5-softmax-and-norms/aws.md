# P5.5 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4). This module needs ≈ 2 instance-hours. Optionally repeat the bf16 rows on a `g6.xlarge` (L4, native bf16 arithmetic on sm_89).

```bash
for e in 01_softmax_ladder 02_rmsnorm 03_fused_residual_rmsnorm 04_fp16_bf16; do ./build/p5/p5.5_$e; done | tee results/p5.5.txt
for x in 01-online-softmax 02-rmsnorm 04-cross-entropy; do ./build/p5/p5.5_$x --bench; done
ctest --test-dir build/d4 -R softmax --output-on-failure
```

Teardown and auto-stop: see the shared page.
