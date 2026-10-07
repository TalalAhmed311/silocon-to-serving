# Exercise 4 — The cost model (T0)

Read `platform/cost/model.py`. `test_cost.py` checks it with round, made-up prices (they are inputs, so the test can't go stale):

- two tenants on one replica are charged by token share
- a TP=2 replica costs 2 GPU-hours per hour
- a 1/7 MIG slice costs 1/7 of a GPU-hour
- a spot node at 40% of the on-demand price plus 5% replay overhead
- idle waste is computed against a capacity, and breakeven utilization is checked against an API price

**Then:** write `platform/cost/prices.yaml` with **your region's real prices**, recording the date and the source page in a comment, and compute your P3.5 "predict first" numbers with `model.py`.
