# Exercise 3: 2-stage → 3-stage `cp.async` pipeline (sm_80+, T2)

`d4::hgemm_mma_async<STAGES>` overlaps the global→shared copy of later K-tiles with the MMAs on the current one. `cp.async` copies without passing through registers, and `cp.async.wait_group N` waits until at most N groups are still in flight.

1. Read the prologue and main loop in `d4/hgemm.cuh`. On paper, for `STAGES = 3`, list which tile each buffer holds at each iteration, and show that the `__syncthreads()` after the wait is what makes overwriting buffer `(t − 1) % S` safe.
2. Run `hgemm_bench` on an L4 and record `mma.sync` (no pipeline) vs 2 stages vs 3 stages at N = 2048, 4096, 8192.
3. Profile two of them with ncu: `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` should fall as stages go up. What's the cost? (Shared memory per block → occupancy. Check the Occupancy section.)
4. Try `STAGES = 4`. Does it still help on your GPU?

`TODO(run-on: g6.xlarge)`
