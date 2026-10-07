# P5.4 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4). This module needs ≈ 2 instance-hours.

```bash
for e in 01_reduce_ladder 02_warp_reduce 03_scan; do ./build/p5/p5.4_$e; done | tee results/p5.4.txt
./build/p5/p5.4_01-reduce-ladder --bench && ./build/p5/p5.4_03-scan --bench
sudo $(which ncu) -k regex:reduce_r[23] --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/p5/p5.4_01_reduce_ladder
```

Teardown and auto-stop: see the shared page.
