# P1.5 exercises

Implement the functions in [`calculator.py`](calculator.py). Each one corresponds to a row of the lesson's formula table. The tests in `test_calculator.py` are the spec.

1. **Weights bytes.** `weights_bytes(cfg, dtype)` must equal, to the byte, the summed tensor sizes of a checkpoint written by `platform/engine/v0/tools/make_tiny_llama.py`, with and without `--tie`. Then, by hand (`TODO(run)`, needs the Hub): for 3 real models, compare with the sizes of their `*.safetensors` files and record the % difference in your notes. Expect < 2%: the extra bytes are headers and optional tensors.
2. **KV and capacity.** `kv_bytes_per_token(cfg, kv_dtype)` and `max_sequences(cfg, gpu, weight_dtype, kv_dtype, context, mem_util, activation_gb)`.
3. **vs vLLM (T2).** `parse_vllm_kv_tokens(log_text)` extracts the KV-cache token capacity vLLM logs at start-up. The test uses a sample line in the format the author expects. **The exact log format is UNVERIFIED**: if your vLLM prints something else, fix the regex and the sample. Then, on a `g6.xlarge` (`TODO(run-on: g6.xlarge)`), compare it with `max_sequences × context` from your calculator and explain the gap.
4. **Cross-check (manual).** Fill in [`crosscheck.md`](crosscheck.md).
5. **MoE + TP.** `total_params` and `active_params` for MoE configs, plus `tp` support in `weights_bytes` and `kv_bytes_per_token`.
