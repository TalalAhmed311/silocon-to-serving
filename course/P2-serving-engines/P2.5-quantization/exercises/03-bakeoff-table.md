# Exercise 3 — The #7 table (T0, then T2)

1. **T0.** `test_table.py` builds `platform/bakeoff/table.py`'s table from fixture results: two variants with fake load JSONs and fake lm-eval outputs. Make it pass with the given code, then add a **"decode speedup vs bf16"** column and a **"bits/weight"** column (read from `variants.yaml`, or inferred from the name). Update the test so that it covers both.
2. **T2.** Run the real bakeoff ([aws.md](../aws.md)) and paste the table into `results/p25_table.md`. Write three conclusions, each tied to a number, and one caveat about statistical power, using `limit=250`.
