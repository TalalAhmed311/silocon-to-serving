# Exercise 3: busbw from algbw (T0)

nccl-tests prints two bandwidths. `algbw = S / t` is what your code feels. `busbw = algbw × factor` is comparable to the link's peak whatever the collective and the rank count. The factors are in `commmodel.BUSBW_FACTOR`, taken from nccl-tests `doc/PERFORMANCE.md` at v2.21.1.

1. Derive the all-reduce factor `2(n−1)/n` from the ring's byte count (exercise 1). `test_busbw_of_modelled_ring_is_link_bandwidth` checks it.
2. Derive `(n−1)/n` for all-gather. Why is broadcast's factor 1?
3. **T3:** for your nccl-tests runs, take the plateau busbw of all_reduce and all_gather. Are they close? (They should be. If not, which topology effect explains it? Read `topo.txt`.)
