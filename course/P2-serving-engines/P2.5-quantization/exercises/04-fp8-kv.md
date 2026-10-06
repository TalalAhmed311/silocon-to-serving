# Exercise 4 — FP8 KV cache: capacity vs quality (T2)

Compare `bf16` with `bf16+fp8kv` from the bakeoff, then answer with numbers in `results/p25_fp8kv.json`:

```json
{"kv_tokens_bf16": 0, "kv_tokens_fp8": 0, "knee_bf16": 0.0, "knee_fp8kv": 0.0,
 "score_bf16": 0.0, "score_fp8kv": 0.0, "long_context_score_bf16": null, "long_context_score_fp8kv": null}
```

1. Is the KV-token ratio ≈ 2? If not, why? (Hint: per-block scales, the activation reserve.)
2. Did the knee move? Explain it with P1.2 §3: at what context lengths does it matter most?
3. Quality: short-context tasks rarely show a KV-quantization effect. Add one long-context task, if you can find a suitable one in lm-eval at 0.4.13, and say whether it does.

`uv run python course/P2-serving-engines/P2.5-quantization/exercises/check_fp8kv.py results/p25_fp8kv.json`
