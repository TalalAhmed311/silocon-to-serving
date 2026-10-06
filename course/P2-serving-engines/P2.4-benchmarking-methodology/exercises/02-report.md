# Exercise 2 — The report generator (T0)

Read `examples/report.py`, then extend `render()` so that it:

1. adds a **variance** column when the input JSON contains repeated runs (a list of `rows` lists): report the median, plus the min–max of p90 TTFT across repeats
2. prints a warning when any row has n < 100: "percentiles above p90 are unreliable at this n"
3. refuses to render if `--hardware` doesn't mention a GPU or CPU model. A free-text "my machine" isn't reproducible

**Test (`test_report.py`):** the required-field check (already passing with the given code), plus your three additions. The test for each addition is skipped until you implement it: look for `pytest.skip` in the test and delete that line once you're done.
