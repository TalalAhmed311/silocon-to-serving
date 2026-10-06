# Exercise 1 — Make D3 match the startup log within 5% (T2)

1. Start vLLM with the defaults (`examples/01_serve.sh`) and run `examples/04_d3_vs_log.py`.
2. The ratio "reported / predicted" will likely be off by 5–20%. Find out why. Candidate reasons:
   - vLLM's activation-profiling peak is larger or smaller than D3's `activation_gb`
   - CUDA-graph memory
   - the non-PyTorch CUDA context (~0.3–0.5 GiB)
   - block-size rounding
3. Adjust **one** assumption at a time until you are within 5%. Then change `--gpu-memory-utilization` to 0.80 and check that the prediction still holds without re-tuning. A calibration that only fits one point isn't a model.
4. Record it in `results/p21.json` (schema in `check_results.py`):

```json
{"gpu": "L4", "model": "<repo>@<sha>", "vllm_version": "0.31.0",
 "kv_tokens_reported": 0, "kv_tokens_predicted": 0, "activation_gb": 1.0, "mem_util": 0.9,
 "decode_tok_s_batch1_graphs": 0.0, "decode_tok_s_batch1_eager": 0.0, "ceiling_tok_s_batch1": 0.0,
 "client_tok_s_batch1": 0.0, "vllm_bench_latency_tok_s": 0.0}
```
