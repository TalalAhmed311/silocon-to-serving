# P5.3 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4), including the Nsight counter section. This module needs ≈ 2 instance-hours.

```bash
sudo $(which ncu) --set full -o results/planted ./build/p5/p5.3_01_planted_slow
nsys profile -o results/nvtx ./build/p5/p5.3_02_nvtx_ranges
sudo $(which ncu) --csv --metrics gpu__time_duration.sum,dram__bytes.sum ./build/p5/p5.2_02_transpose_ladder > results/transpose.csv
tar czf - results/*.ncu-rep results/*.nsys-rep | base64 > results/reports.b64     # copy this text back, decode locally
```

Teardown and auto-stop: see the shared page.
