# P5.7 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g6.xlarge`** (L4, sm_89). `mma.sync`, `ldmatrix` and `cp.async` need sm_80+. WMMA alone also runs on a `g4dn.xlarge`. This module needs ≈ 3 instance-hours.

```bash
./build/p5/p5.7_hgemm_bench | tee results/p5.7-hgemm.md
./build/p5/p5.7_int8_gemm
./build/p5/p5.7_01-wmma --bench && ./build/p5/p5.7_02-mma-sync --bench
for k in hgemm_mma hgemm_mma_async; do sudo $(which ncu) -k regex:$k -c 2 --set full -o results/$k ./build/p5/p5.7_hgemm_bench; done
```

Teardown and auto-stop: see the shared page.
