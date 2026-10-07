# Exercise 3: remove the divergence from a planted kernel (T2)

`kernel.cuh` clamps and scales with three lane-dependent branches. On random data, every warp takes all three paths, serially, with lanes masked off.

1. Run the test with `--bench` and record the time. Profile with `ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio` (average active threads per executed instruction: 32 means no divergence).
2. Rewrite it **branch-free** with identical results (the test requires exact equality), re-run, and re-profile.
3. When is divergence harmless? (Short branches become predicated selects, and conditions that are uniform across a warp cost nothing. See `examples/03_divergence.cu`.)
