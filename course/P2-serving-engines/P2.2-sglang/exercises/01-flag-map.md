# Exercise 1 — Flag-mapping table, verified by `--help` (T2 to capture, T0 to test)

1. On the instance, capture both help texts **at the pinned versions**:
   ```bash
   .venv-serving/bin/vllm serve --help=all > course/P2-serving-engines/P2.2-sglang/exercises/help/vllm-0.31.0.txt 2>&1 || \
     .venv-serving/bin/vllm serve --help > course/P2-serving-engines/P2.2-sglang/exercises/help/vllm-0.31.0.txt
   .venv-sglang/bin/python -m sglang.launch_server --help > course/P2-serving-engines/P2.2-sglang/exercises/help/sglang-0.5.21.txt
   ```
2. Commit them. `test_flag_map.py` checks that every flag in [`flag_map.yaml`](flag_map.yaml) appears in the right help file, so a renamed flag fails the test and not your deployment.
3. Add 3 rows of your own: flags you'll need in P2.3–P2.6.

Until the help files exist, the test is **skipped** (it reports "capture the help text first").
