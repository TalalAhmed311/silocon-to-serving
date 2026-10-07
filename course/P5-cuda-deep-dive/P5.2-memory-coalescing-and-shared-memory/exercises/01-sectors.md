# Exercise 1: predict sectors per request, then confirm with ncu (T2)

A warp-wide 4-byte load touches some set of **32-byte sectors**. Predict sectors per request for each pattern, then measure with:

```bash
sudo $(which ncu) --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  ./build/p5/p5.2_01_strided_read
```

| pattern | your prediction | measured (sectors ÷ requests) |
|---|---|---|
| `in[i]`, aligned (stride 1, offset 0) | | |
| `in[i + 1]` (misaligned by 4 B) | | |
| `in[2 * i]` (stride 2) | | |
| `in[32 * i]` (stride 32) | | |

Then explain the GB/s column of `01_strided_read` from your sector counts: useful bytes ÷ fetched bytes. `TODO(run-on: g4dn.xlarge)`
