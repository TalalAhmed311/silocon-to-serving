# Exercise 1: online softmax, reported in GB/s (T2, L4 exit check)

Implement `softmax_rows` in `kernel.cuh`. Climb, measuring each step:

1. **Safe softmax**, 3 passes: max, then Σ exp(x − max), then normalize. Three reads of the row.
2. **Online softmax**: keep `(m, d)` = running max and running Σ exp(x − m). On a new value `v > m`, rescale: `d ← d·exp(m − v) + 1; m ← v`. Merging two partial states (across lanes and warps) is `m = max(m₁, m₂); d = d₁·e^{m₁−m} + d₂·e^{m₂−m}`. Two reads.
3. **Single read**: one warp per row, the row held in registers (32 values per lane covers 1024 columns). One read + one write: the minimum.

The test includes logits in [500, 1000], where `exp(x)` without the max overflows. `--bench` reports GB/s (1 read + 1 write counted) for 1024 and 32000 columns. That GB/s figure is the L4 exit check deliverable. `TODO(run-on: g4dn.xlarge)`
