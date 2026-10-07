# P2.1 — First serve with vLLM

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) one NVIDIA GPU with ≥ 24 GB (AWS `g6.xlarge` by default, see [aws.md](aws.md)). The client and analysis scripts are T0 and can be tested against `platform/mockllm` |
| **Time** | ≈25 min reading + ≈7 h hands-on (≈3–4 GPU-hours) |
| **Prerequisites** | P1.2 (prefill/decode), P1.3 (metrics), P1.4 (roofline), P1.5 (your capacity predictions for this exact setup) |
| **Pinned** | vLLM `v0.31.0` (`vllm-project/vllm@d1f3d8b8`), installed from `env/requirements-serving.txt` |

## Learning objectives

1. Serve a model with `vllm serve` and query its OpenAI-compatible API, streaming and not.
2. Read the startup log (weights loaded, KV-cache blocks allocated, CUDA graphs captured) and **reconcile it with your D3 prediction**.
3. Know the flags that matter, and what each one trades.
4. Measure batch-1 decode tokens/s and explain the gap to the P1.4 ceiling term by term.

## Why this matters

From here on every number comes from a real engine. vLLM is the most widely deployed open-source LLM server, and its flags map one-to-one to the concepts of P1:

- `--max-model-len`: KV per sequence
- `--gpu-memory-utilization`: the KV budget
- `--max-num-seqs`: batch size
- `--max-num-batched-tokens`: the prefill chunk size
- `--kv-cache-dtype`: bytes per KV element

---

## 1. Model choice and licensing

Pick a **7–8B instruct model** you are licensed to use (Llama-3.x-8B-Instruct, Qwen2.5-7B-Instruct and Mistral-7B-Instruct are common choices). Accept its license on the Hub if it is gated, and **pin a revision SHA**, so that `bootstrap.sh <repo> <sha>` downloads exactly the weights your numbers refer to. Record the repo and SHA in your notes. For the cheapest path (`g4dn.xlarge`, T4 16 GB), use a ≤ 3B model: a 7B model in fp16 needs about 14 GiB for weights alone.

## 2. The minimal server

```bash
vllm serve /opt/models/<model> --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --api-key "$S2S_API_KEY"
```

- `--host 127.0.0.1` together with `--api-key` means no unauthenticated endpoint, even by accident. You reach it with `make forward` (SSM port-forward).
- The startup log reports the weights' memory, then profiles a forward pass to measure activation memory, then sizes the KV cache from what is left. It prints the KV capacity in **tokens** and the maximum concurrency at `--max-model-len`. Then it captures CUDA graphs for a set of batch sizes. The exact wording changes between versions: `examples/04_d3_vs_log.py` greps for it and tells you if the pattern didn't match.

## 3. The flags that matter

| Flag | Concept (lesson) | Turning it up… |
|---|---|---|
| `--max-model-len` | context length per sequence (P1.2) | longer prompts, fewer concurrent sequences, and possibly startup failure if one sequence's KV doesn't fit |
| `--gpu-memory-utilization` | KV budget = this × VRAM − weights − activations (P1.5) | more KV, but less headroom (other processes, fragmentation) |
| `--max-num-seqs` | max running batch (P1.2 §3) | more throughput, until KV runs out or ITL rises past your SLO |
| `--max-num-batched-tokens` | per-step token budget, i.e. the chunked-prefill size (P2.3) | faster TTFT for long prompts, but spikier ITL for everyone else |
| `--enable-prefix-caching` | reuse KV of shared prefixes (P2.3) | big TTFT wins for repeated system prompts. It is on by default in recent V1 versions: check `--help` |
| `--enforce-eager` | disable CUDA graphs (P2.6) | slower decode at small batch (launch overhead), but less memory and faster startup |
| `--kv-cache-dtype fp8` | 1 byte per KV element (P1.2) | ~2× KV capacity, with a quality impact to measure (P2.5) |
| `--quantization …` | weight format (P2.5) | fewer weight bytes, faster memory-bound decode |
| `--tensor-parallel-size N` | shard across N GPUs (P4.2) | fits bigger models, at the cost of an all-reduce per layer |

Always check `vllm serve --help` for *your* pinned version. Defaults move between releases.

## 4. Measuring decode against the ceiling

> **Predict first.** From P1.4: Llama-3-8B bf16 on an L4 has a batch-1 decode ceiling of ≈ 18.7 tok/s (≈ 53 ms/token). Your D3 (P1.5) run with `bw_util = 0.8` predicts ≈ 15 tok/s. Write your number down before you run `bench/decode_ceiling.py`.

`bench/decode_ceiling.py` sends one streaming request at a time with a short prompt and a long output, and reports the median ITL over the decode phase. Then it does the same at batch 2, 4 and 8, with 8 concurrent streams. Compare the measurement with the ceiling and explain the gap, term by term:

| Term | How to estimate it |
|---|---|
| achievable vs peak bandwidth | P1.4 exercise 4 (your measured copy GB/s ÷ spec) |
| KV-cache reads | grow with position: rerun with a 4k prompt |
| non-GEMM kernels, launch gaps | compare `--enforce-eager` with the default (CUDA graphs) |
| sampling + detokenization + HTTP | client-side ITL vs `vllm:inter_token_latency_seconds` (P1.3) |

---

## Walkthrough (on the instance; see [aws.md](aws.md))

```bash
bash infra/aws/scripts/bootstrap.sh p2 <model-repo> <revision-sha>
source .venv-serving/bin/activate
bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model>  # leaves the server running
python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/02_client.py --url http://127.0.0.1:8000
python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/03_offline_llm.py --model /opt/models/<model>
python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/04_d3_vs_log.py --log vllm.log --model-config /opt/models/<model>/config.json --gpu L4
python course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py --url http://127.0.0.1:8000 --gpu L4 --model-config /opt/models/<model>/config.json
```

On a laptop, try every client script against the mock first: `uv run uvicorn mockllm.server:app --app-dir platform --port 8000`.

## What you should see

`TODO(run-on: g6.xlarge)` for every number. Expected shapes:

- `01_serve.sh` writes `vllm.log`. `04_d3_vs_log.py` prints `predicted KV tokens | vLLM KV tokens | ratio`, which should agree within ~10–20%. The gap is your calibration of D3's `activation_gb` and `mem_util`.
- `02_client.py` prints a completion, then TTFT and ITL for a streamed request.
- `decode_ceiling.py` prints `batch | measured tok/s per seq | aggregate | predicted ceiling | % of ceiling`. Batch 1 lands at ~70–90% of the ceiling. Aggregate throughput rises almost linearly with batch at short context, because decode is memory-bound and the weights are shared (P1.2 §3).

## Exercises

| # | Exercise | Tier | Check |
|---|---|---|---|
| 1 | [Make D3 match the startup log within 5%](exercises/01-calibrate-d3.md) | T2 | `exercises/check_results.py` validates your recorded `results/p21.json` |
| 2 | [Find the `--max-model-len` that fails, and explain why](exercises/02-max-model-len.md) | T2 | written answer + the log line |
| 3 | [Eager vs CUDA graphs at batch 1](exercises/03-eager-vs-graphs.md) | T2 | `check_results.py` (graphs ≥ eager in tok/s) |
| 4 | [Reproduce `vllm bench latency` with your own client](exercises/04-reproduce-bench.md) | T2 (T0 test vs mock) | `test_client_vs_mock.py` (T0); within 10% of `vllm bench` (T2) |

```bash
uv run pytest course/P2-serving-engines/P2.1-first-serve-with-vllm/exercises     # T0 parts
uv run python course/P2-serving-engines/P2.1-first-serve-with-vllm/exercises/check_results.py results/p21.json
```

## Common mistakes

- Binding to `0.0.0.0` "just to test". Don't. Use the SSM port-forward.
- Benchmarking the **first** request: it includes CUDA-graph capture warm-up and torch.compile caches. Always warm up.
- Comparing tok/s across runs with different prompt lengths, output lengths or sampling settings.
- Forgetting that `max_tokens` caps the output. A short answer measures TTFT, not decode.
- Running out of disk: an 8B bf16 model is ~16 GB, plus the vLLM wheels and caches. The default 200 GB volume is plenty, but check with `df -h`.

## Go deeper

- vLLM docs (`docs/` at `vllm-project/vllm@d1f3d8b8`): quickstart, the OpenAI-compatible server, engine arguments, `docs/benchmarking/cli.md`, `docs/design/metrics.md`.
- Modular handbook: *Choosing the right inference framework*.
- Silicon to Scale ch. 6 (inference systems).
- Inference Engineering Academy, vLLM engine course: **UNVERIFIED** (the site could not be checked from the build environment; see GAPS.md).

## Go down when…

Decode is far below the ceiling → profile it (P5.3), then P6 (engine internals). **Next:** [P2.2 SGLang](../P2.2-sglang/README.md).
