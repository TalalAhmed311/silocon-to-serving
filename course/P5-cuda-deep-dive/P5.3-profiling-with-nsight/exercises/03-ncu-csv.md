# Exercise 3: `ncu --csv` → the bench table's "% of peak" column (T0 + T2)

`platform/kernels/bench/ncu_to_table.py` turns per-kernel ncu metrics into `| kernel | time | GB/s | % of copy | TFLOP/s | % of peak |`. `test_ncu_to_table.py` checks it on a synthetic CSV (the format, not real numbers).

1. Make the test pass (it should as shipped), and read how units and thousands separators are handled.
2. On T2: `sudo ncu --csv --metrics gpu__time_duration.sum,dram__bytes.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum ./build/p5/p5.2_02_transpose_ladder > results/transpose.csv`, then run the script with `--peak-gbs` from `s2s::measure_copy_gbs()` (printed by the harness). Do ncu's DRAM bytes agree with the 8·R·C bytes the harness assumes? Explain any difference (L2 hits, write-allocate).
3. Add a column for **achieved occupancy** (`sm__warps_active.avg.pct_of_peak_sustained_active`) and a test for it.

Commit the CSV summaries next to your benches (the `.ncu-rep` files are git-ignored: they're large and machine-specific).
