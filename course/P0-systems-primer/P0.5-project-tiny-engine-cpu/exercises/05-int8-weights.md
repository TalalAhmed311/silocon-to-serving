# Exercise 5 — Int8 weights, `runq`-style (hard)

llama2.c's `runq.c` stores weights as int8 with one fp32 scale per group, and quantizes activations on the fly. That cuts the weight bytes 4×. P0.5 §2 says decode is bandwidth-bound, so up to ~4× more tokens/s should follow.

**Part A (tested).** Implement `quantize_matrix` and `matvec_q8` in [`05-int8/q8_matvec.hpp`](05-int8/q8_matvec.hpp) (32-element blocks, symmetric, scale = max|w| / 127). `ctest -R 05` checks every output against the derived bound `Σ |w|·δx + |x̂|·δw`, with δ = scale/2.

**Part B (measured, not auto-tested).** Wire it into a copy of `engine.hpp`:
- quantize every 2-D weight at load time (or offline, into an `I8` + `F32` scales safetensors file)
- replace `ops::matvec` with `matvec_q8`, threaded with the pool

Then report:

| model | fp32 decode tok/s | q8 decode tok/s | speedup | weight MiB fp32 → q8 | mean |Δlogit| teacher-forced |
|---|---|---|---|---|---|

If you convert a real model, also report perplexity on a fixed text before and after. `reference/llama_numpy.py` plus a 20-line loop computing `exp(mean(−log p(next)))` is enough.

**Expected shape:** a large speedup for models whose fp32 weights don't fit in cache, and almost none for the tiny test model, which is not DRAM-bound. Logit differences of order 1e-2 should give a perplexity change of a few %.

**Reference:** [`solutions/q8_matvec.hpp`](solutions/q8_matvec.hpp) for part A. Part B is yours: it is the prototype of the quantized path you'll benchmark properly on GPUs in P2.5.
