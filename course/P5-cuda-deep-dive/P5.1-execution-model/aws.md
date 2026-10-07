# P5.1 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4). This module needs ≈ 2 instance-hours.

```bash
for e in 01_hello_indices 02_occupancy 03_divergence 04_pinned_vs_pageable 05_streams_overlap; do ./build/p5/p5.1_$e; done | tee results/p5.1.txt
./build/p5/p5.1_01-vector-add --bench && ./build/p5/p5.1_03-divergence --bench
sudo $(which ncu) --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./build/p5/p5.1_03-divergence --bench
nsys profile -o results/overlap ./build/p5/p5.1_05_streams_overlap
```

Teardown and auto-stop: see the shared page.
