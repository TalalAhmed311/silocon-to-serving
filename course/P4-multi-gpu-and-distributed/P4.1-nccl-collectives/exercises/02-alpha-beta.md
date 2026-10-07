# Exercise 2: fit the α–β model to nccl-tests output (T0 with synthetic data → T3 with yours)

`commmodel.fit_alpha_beta(sizes, times)` fits `t = a + b·S`. For a ring all-reduce, `a = 2(n−1)·α` and `1/b = n/(2(n−1)) · B`.

1. `test_comm_model.py::test_fit_recovers_alpha_beta` generates noisy times from known α and B and checks the fit recovers them. Read it.
2. **T3:** run `examples/03_nccl_tests.sh`, parse with `bench/parse_nccl_tests.py --json`, and fit **only** the large sizes (≥ 1 MB) for B and **only** the small sizes (≤ 64 KB) for α. Why does fitting all sizes at once give a worse α?
3. Write down α (µs) and B (GB/s) for each instance you ran. `TODO(run-on: g6.12xlarge, p4d.24xlarge)`. They feed P4.2's TP cost prediction.
