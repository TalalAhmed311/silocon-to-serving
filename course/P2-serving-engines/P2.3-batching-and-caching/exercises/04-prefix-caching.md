# Exercise 4 — Prefix caching on vLLM (hard, T2)

Follow [aws.md](../aws.md): run `loadgen.cli` with `--shared-prefix 1000 --shared-fraction 0.8` against vLLM with prefix caching on, then again with it off (`--no-enable-prefix-caching`). Then:

1. Plot both with `loadgen.plot` and save `results/knee.png`.
2. Record `results/p23.json`:
   ```json
   {"gpu": "L4", "model": "<repo>@<sha>", "knee_on_rps": 0.0, "knee_off_rps": 0.0,
    "ttft_p50_on_s_at_low_rate": 0.0, "ttft_p50_off_s_at_low_rate": 0.0, "prefix_hit_rate": 0.0}
   ```
   `prefix_hit_rate`: read it from vLLM's `/metrics` (find the prefix-cache hit and query counters at your pinned version; their names are in `docs/usage/metrics.md`).
3. Run `uv run python course/P2-serving-engines/P2.3-batching-and-caching/exercises/check_prefix.py results/p23.json`.
4. Explain the TTFT difference with numbers: 1000 cached tokens × your measured per-token prefill cost, from P2.1 or the D3 prefill estimate.
