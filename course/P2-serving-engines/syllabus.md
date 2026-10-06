# P2 — Serving with inference engines

**Weeks 7–12 · Lane A ≈ 66 h · Tier T2 (one NVIDIA GPU: local, or AWS `g6.xlarge`/`g5.xlarge`) with T0 parts (gateway, analysis) · Lane B: L2 (weeks 7–14)**

**Thread through the phase:** run vLLM and SGLang, push them to KV-cache saturation, find the latency knee, then change one thing at a time (batching policy, quantization, speculative decoding, CUDA graphs) and measure what it bought. Projects #4, #7, #10 and #6 come out of this phase.

Pinned engines: vLLM `v0.31.0` (`d1f3d8b8`), SGLang `v0.5.21` (`a34cba3c`), GuideLLM `v0.8.0`, LLM Compressor `0.14.0`. See SOURCES.md.

**Default model and GPU.** A 7–8B instruct model (license checked and pinned by revision at build time) on an L4 24 GB (`g6.xlarge`). A ≤1B model on T4 (`g4dn.xlarge`) is the budget path. No FP8 or bf16 on T4, which the lessons call out.

| Module | Time | Project | Animations |
|---|---|---|---|
| P2.1 First serve with vLLM | 0.4 h + 7 h | — | — |
| P2.2 SGLang | 0.3 h + 6 h | — | — |
| P2.3 Batching and caching | 0.6 h + 12 h | **#4 continuous batching load test** | **static vs continuous batching; chunked prefill letting decodes continue**; **PagedAttention logical→physical blocks, prefix sharing, copy-on-write**; handbook BatchingSimulator, ChunkedPrefillVisualizer (linked) |
| P2.4 Benchmarking methodology | 0.4 h + 6 h | (harness used by #4, #7, #10) | — |
| P2.5 Quantization | 0.6 h + 12 h | **#7 quantized serving bakeoff** | — |
| P2.6 Speculative decoding and CUDA graphs | 0.6 h + 12 h | **#10 spec-decode prototype** | **speculative decoding: draft proposes k, target verifies, accept/reject** |
| P2.7 Gateway basics | 0.4 h + 10 h | **#6 multi-model AI gateway** (T0 core) | — |

Each T2 module has an `aws.md`: `g6.xlarge`, cost range (checked in-region at build time), launch via `infra/aws/single-node`, run, `make down`, idle auto-stop, budget alarm, endpoint bound to `127.0.0.1` and reached through an SSM port-forward.

---

## P2.1 — First serve with vLLM

**Objectives.** (1) Serve a model with `vllm serve` and query its OpenAI-compatible API. (2) Read the startup log: weights loaded, KV-cache blocks allocated, CUDA-graph capture. Reconcile it with your D3 calculator. (3) Know the flags that matter: `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, `--max-num-batched-tokens`, `--tensor-parallel-size`, `--enable-prefix-caching`, `--enforce-eager`, `--quantization`, `--kv-cache-dtype`. (4) Measure batch-1 decode tok/s vs the P1.4 prediction.

**Examples.** `01_serve.sh` · `02_client.py` (streaming, measures TTFT/ITL with P1.3 code) · `03_offline_llm.py` (`LLM.generate` batch API) · `04_d3_vs_log.py`.

**Exercises.** (1) Make D3's KV capacity prediction match the startup log within 5% · (2) find the `--max-model-len` that OOMs and explain why · (3) measure eager vs graphs at batch 1 · (4) *(hard)* run `vllm bench latency` and reproduce its number with your own client.

**Bench.** `bench/decode_ceiling.py`: `batch | measured tok/s | predicted ceiling | % of ceiling`. `TODO(run-on: g6.xlarge)` until run.

**Sources.** vLLM `docs/` (quickstart, serving, engine args), `vllm/benchmarks/` · Inference Engineering Academy engine course (UNVERIFIED; linked once verified) · Silicon to Scale ch 6.

## P2.2 — SGLang

**Objectives.** (1) Serve the same model with `python -m sglang.launch_server`. (2) Compare flag-for-flag with vLLM (mem fraction, chunked prefill size, radix cache). (3) Run the same client and bench, then compare in one table. (4) Read `python/sglang/srt/managers/scheduler.py` at a high level: where a batch is formed.

**Examples.** `01_serve.sh` · `02_same_client.py` · `03_frontend_lang.py` (SGLang's structured generation primitives, briefly).

**Exercises.** (1) Flag-mapping table, auto-checked against `--help` output stored at the pinned version · (2) multi-turn chat workload where the radix cache wins: measure the cache hit rate · (3) *(hard)* find one workload where vLLM wins and explain why.

**Sources.** SGLang `docs/`, `python/sglang/srt/` · RadixAttention paper (2312.07104).

## P2.3 — Batching and caching (+ #4 load test)

**Objectives.** (1) Explain static, dynamic and continuous (iteration-level) batching. (2) Explain chunked prefill and why it protects ITL. (3) Explain PagedAttention blocks and prefix caching (block hashing). (4) **Build #4:** a traffic generator that ramps request rate or concurrency until the KV cache saturates (watch `vllm:kv_cache_usage_perc` and preemptions) and locates the latency knee.

**Examples.** `01_batching_sim.py` (T0 discrete-event simulator: static vs continuous, plots timelines) · `02_chunked_prefill_on_off.sh` · `03_prefix_cache_hit.py` (shared system prompt, hit rate from metrics).

**Exercises.** (1) Implement continuous batching in the simulator (pytest: lower mean latency than static on fixed traces) · (2) add chunked prefill to the simulator, with a test showing bounded ITL · (3) **#4 load test**: open-loop Poisson arrivals, configurable prompt/output length distributions, knee detection (largest rate where p90 TTFT < SLO). Tested against the mock server from P1.3 · (4) *(hard)* reproduce the knee on vLLM with prefix caching on vs off.

**Bench.** #4 prints `rate | concurrency | KV usage % | preemptions | p50/p90 TTFT | p50/p90 ITL | goodput` and saves JSON for a knee plot.

**Animations (required).** `animations/p2-continuous-batching.html`, `animations/p2-paged-attention.html`. Linked: handbook BatchingSimulator, ChunkedPrefillVisualizer.

**Sources.** Orca (OSDI '22) · SARATHI / Sarathi-Serve · PagedAttention (2309.06180) · handbook `static-dynamic-continuous-batching.md`, `pagedattention.md`, `prefix-caching.md` · vLLM `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py` (read-only tour; deep dive in P6).

**Go down when…** the knee is in the wrong place: P6.2–P6.3.

## P2.4 — Benchmarking methodology

**Objectives.** (1) Closed-loop vs open-loop load, and coordinated omission. (2) Warm-up, repetitions, median/p90, and reporting variance. (3) Use GuideLLM and `vllm bench serve`, then reconcile them with #4. (4) Write a benchmark report that someone else can reproduce (hardware, versions, flags, dataset, seed).

**Exercises.** (1) Show coordinated omission with a closed-loop client vs open-loop on the mock server · (2) the report template generator (pytest checks required fields) · (3) GuideLLM sweep vs #4 on the same server: explain any difference.

**Sources.** GuideLLM `src/guidellm/` · vLLM `vllm/benchmarks/`, `docs/benchmarking/` · handbook `llm-performance-benchmarks.md` · Silicon to Scale ch 12.

## P2.5 — Quantization (+ #7 bakeoff)

**Objectives.** (1) Number formats: FP16/BF16/FP8 (E4M3/E5M2)/INT8/INT4. Weight-only vs weight+activation, and KV-cache quantization. (2) How AWQ and GPTQ choose scales, at the level of their papers. (3) Produce AWQ/GPTQ/FP8 checkpoints with LLM Compressor (AutoAWQ is deprecated; see SOURCES.md) and GPTQModel. (4) **#7:** FP16 vs AWQ vs FP8 on the same GPU: quality (lm-eval-harness tasks, pinned), latency, throughput and VRAM.

**Examples.** `01_formats.py` (T0: round-trip error per format, plots) · `02_quantize_llmcompressor.py` · `03_serve_quantized.sh` · `04_lm_eval.sh`.

**Exercises.** (1) Implement symmetric/asymmetric per-group INT4 quant/dequant in NumPy (pytest error bounds) · (2) FP8 E4M3 rounding emulation vs `torch.float8_e4m3fn` (T0, PyTorch CPU) · (3) **#7 bakeoff** harness: one command runs all variants and writes the comparison table · (4) *(hard)* FP8 KV cache: measure the extra concurrency vs the quality delta.

**Bench (#7).** `variant | VRAM GB | max seqs | p50 TTFT | decode tok/s @ batch 1/32 | task scores`. FP8 needs L4 or better (`TODO(run-on: g6.xlarge)`).

**Sources.** AWQ (2306.00978) · GPTQ (2210.17323) · FP8 formats (2209.05433) · LLM Compressor, GPTQModel, lm-evaluation-harness (pinned) · vLLM `vllm/model_executor/layers/quantization/` · handbook `llm-quantization.md` · Silicon to Scale ch 8 · llama.cpp block formats (P0.4 bridge).

## P2.6 — Speculative decoding and CUDA graphs (+ #10)

**Objectives.** (1) Derive the expected speedup from acceptance rate α and draft length k (Leviathan et al.). (2) Draft model vs n-gram vs EAGLE/Medusa-style heads. (3) Implement speculative sampling verification correctly (it preserves the target distribution). (4) **#10:** prototype draft+target in PyTorch with prefix caching and chunked prefill, then compare its acceptance rate and real speedup with vLLM's built-in spec decode. (5) Why CUDA graphs help decode (launch overhead) and what they constrain (static shapes, capture sizes).

**Examples.** `01_spec_math.py` (T0) · `02_spec_sampling_numpy.py` (T0, the exact rejection-sampling rule) · `03_vllm_spec.sh` · `04_graphs_on_off.sh`.

**Exercises.** (1) Verification rule: a statistical test shows the output distribution equals the target's (T0) · (2) expected-tokens-per-step formula vs simulation · (3) **#10 prototype** with real models (T2) · (4) *(hard)* measure α by domain (code vs chat) and explain why it varies.

**Bench (#10).** `method | k | α | tokens/step | ITL | speedup vs baseline`.

**Animation (required).** `animations/p2-speculative-decoding.html`.

**Sources.** Leviathan et al. (2211.17192) · Medusa (2401.10774) · EAGLE (2401.15077) · vLLM `vllm/v1/spec_decode/`, `docs/features/speculative_decoding/`, `docs/design/cuda_graphs.md` · handbook `speculative-decoding.md` · Silicon to Scale ch 13.

**Go down to:** P6.5 (spec verification on GPU), P6.6 (CUDA graph capture).

## P2.7 — Gateway basics (+ #6)

**Objectives.** (1) What a gateway owns: auth, routing, retries with budgets, fallbacks, rate limits, per-tenant token budgets, streaming passthrough. (2) Read LiteLLM's router and Envoy AI Gateway's rate-limit design, then **build your own** (#6) in Python (FastAPI + httpx) across 2–3 backends: the mock server, vLLM and SGLang. (3) Token-bucket vs sliding-window limits, and counting tokens from usage fields.

**Examples.** `01_proxy_min.py` · `02_router.py` (weighted, least-outstanding-requests) · `03_rate_limit.py`.

**Exercises (T0, all against mock backends).** (1) Token bucket (pytest with fake clock) · (2) retry with budget and jitter, never retrying non-idempotent requests after streaming started · (3) fallback on 5xx/timeout · (4) per-tenant token budget with usage accounting · (5) *(hard)* SSE passthrough with mid-stream backend failure handling.

**Bench.** `bench/gateway_overhead.py`: `rps | p50/p99 added latency` vs direct.

**Lives in.** `platform/gateway/` (#6). Uses API keys from env/Secrets Manager. No unauthenticated endpoints.

**Sources.** LiteLLM (router, rate limiting) · Envoy AI Gateway (pinned) · handbook `inference-routing.md`, `openai-compatible-api.md`.
