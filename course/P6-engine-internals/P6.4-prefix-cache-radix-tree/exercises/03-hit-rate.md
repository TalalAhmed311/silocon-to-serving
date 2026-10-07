# Exercise 3: hit rate on real workload shapes (T0)

`bench/hit_rate.py` replays two synthetic traces through both prefix caches and prints hit rates. Extend it:

1. Add a third trace from real data shapes: a multi-turn conversation dataset's turn lengths (e.g. the ShareGPT-style statistics you used in P2.3; synthesise token ids, keep the lengths).
2. Add an eviction budget (cache capacity in tokens) and plot hit rate vs capacity for both schemes.

| trace | capacity | block-hash hit rate | radix hit rate |
|---|---|---|---|
| multi-turn | ∞ | (from the bench) | |
| few-shot | ∞ | | |
| yours | 25 % of working set | | |

Which workloads favour which scheme, and by how much?
