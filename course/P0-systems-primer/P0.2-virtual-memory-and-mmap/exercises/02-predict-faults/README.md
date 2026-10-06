# Exercise 2 — Predict page faults for an access pattern (easy)

A simulated process touches addresses in a freshly `mmap`'d region. Every page faults exactly once, on its **first** touch. Optionally, the kernel does **fault-around**: when it handles a fault on page *p*, it also maps the other pages in the aligned group of `around` pages that contains *p* (Linux's default `fault_around_bytes` is 64 KiB, which is 16 pages of 4 KiB, for file-backed mappings).

Implement `count_faults(addresses, page=4096, around=1) -> int` in `faults.py`. Then **predict before you run** the counts for the three patterns in the test docstrings.

**Test:** `uv run pytest exercises/02-predict-faults`
