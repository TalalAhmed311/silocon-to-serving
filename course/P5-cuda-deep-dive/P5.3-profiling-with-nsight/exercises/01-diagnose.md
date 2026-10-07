# Exercise 1: diagnose three planted slow kernels from ncu reports alone (T2)

`examples/01_planted_slow.cu` has a reference row-sum and `mystery_a`, `mystery_b` and `mystery_c`. Each is slow for **one** reason. **Don't read the source or the answers at the bottom of the file first.**

```bash
sudo $(which ncu) --set full -o results/planted ./build/p5/p5.3_01_planted_slow
```

Copy `planted.ncu-rep` to your laptop and open it in the Nsight Compute UI. For each mystery kernel, write:

| kernel | Speed-of-Light: memory % / compute % | the section that gave it away | the metric value | diagnosis | fix |
|---|---|---|---|---|---|
| mystery_a | | | | | |
| mystery_b | | | | | |
| mystery_c | | | | | |

Useful places: *Memory Workload Analysis* (sectors/request, L1/L2 hit rates), *Occupancy* (theoretical vs achieved, limiter), *Warp State Statistics* (top stall reason), *Source* view (the line with the most stall samples).

Then implement your fix for each, re-profile, and show the metric that moved. `TODO(run-on: g4dn.xlarge)`
