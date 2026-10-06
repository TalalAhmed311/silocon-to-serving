# Exercise 3 — #10 with a KV cache, and vs vLLM (T0 test + T2 run)

The prototype in `platform/specdec/hf_models.py` recomputes the whole prefix on every call. Add a KV cache:

1. Keep `past_key_values` for the accepted prefix in both the draft and the target.
2. After verification, **truncate** the target's cache to the accepted length. Rejected positions must not leak into the next round. The `DynamicCache` in transformers has a crop operation; check its API at your installed version.
3. **T0 test:** `test_cached_equivalence.py` checks that greedy outputs with your cached models equal the uncached prototype's on tiny random models. It is skipped until you add `HFModelCached` to `hf_models.py`.
4. **T2:** fill in a table on your GPU for the same prompts, `temperature=0.8` and batch 1:

| method | k | α | ITL p50 ms | speedup vs target-only |
|---|---|---|---|---|
| target only (vLLM) | — | — | | 1.0× |
| vLLM n-gram | 4 | | | |
| vLLM draft model | 4 | | | |
| #10 (your cached prototype) | 4 | | | |

5. Explain why your prototype's *speedup* can be closer to vLLM's than its absolute ITL is.
