# P5.2 on AWS

Follow [the shared P5 setup](../aws-common.md) on a **`g4dn.xlarge`** (T4). This module needs ≈ 2 instance-hours.

```bash
for e in 01_strided_read 02_transpose_ladder 03_bank_conflicts; do ./build/p5/p5.2_$e; done | tee results/p5.2.txt
for x in 02-transpose 03-bank-conflicts 04-conv2d-halo; do ./build/p5/p5.2_$x --bench; done
sudo $(which ncu) --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./build/p5/p5.2_01_strided_read
sudo $(which ncu) --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./build/p5/p5.2_03_bank_conflicts
```

Teardown and auto-stop: see the shared page.
