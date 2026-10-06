# P1 — How LLM inference works + hardware math

**Weeks 4–6 · Lane A ≈ 33 h · Tier T0 (all calculators and reference code run on a laptop; one optional T2 measurement) · Lane B: L1 continues**

**Thread through the phase:** by the end you can take any (model, GPU, workload) triple and predict its VRAM use, KV-cache headroom, max batch, TTFT and decode tokens/s from first principles, then explain why a real server misses that prediction. Every number for a GPU comes from a cited NVIDIA whitepaper (see SOURCES.md §4, currently UNVERIFIED), never from memory.

| Module | Time | Project / drill | Animations |
|---|---|---|---|
| P1.1 Transformer forward pass | 0.6 h + 6 h | — | handbook RequestLifecycle (linked) |
| P1.2 Prefill vs decode, KV cache | 0.5 h + 6 h | — | **prefill vs decode token flow with a growing KV cache; TTFT/ITL timeline**; handbook AutoregressiveDecodeStepper, ContextWindowSimulator (linked) |
| P1.3 Latency and throughput metrics | 0.4 h + 5 h | — | handbook LatencyTimelineVisualizer (linked) |
| P1.4 GPU architecture and roofline | 0.6 h + 6 h | — | **roofline** (GPU presets); handbook SMFloorplan, GPUTable (linked) |
| P1.5 Project: capacity calculator | 0.3 h + 9 h | **D3** | handbook GPU memory calculator (linked, used as a cross-check) |

---

## P1.1 — Transformer forward pass, by the numbers

**Objectives.**

1. Write a Llama-style decoder forward pass in NumPy: RMSNorm, RoPE, GQA attention, SwiGLU.
2. Count the parameters, FLOPs and bytes moved for each op, as a function of (d_model, n_layers, n_heads, n_kv_heads, d_ff, vocab).
3. Check your NumPy model against HF `transformers` on a tiny random-init Llama config (CPU).
4. Explain why ≈2·P FLOPs per token is a good approximation and when it breaks (long context: attention FLOPs grow with sequence length).

**Examples.** `01_numpy_llama.py` (the same reference as P0.5, now annotated with FLOP and byte counters) · `02_hf_parity.py` (load a random-init `LlamaForCausalLM` from a config, compare logits) · `03_flop_counter.py`.

**Exercises.** (1) Parameter count formula vs `sum(p.numel())` · (2) FLOPs per token formula vs the instrumented counter · (3) GQA: derive the KV-head saving and test it · (4) *(hard)* RoPE: prove the relative-position property numerically (pytest).

**Bench.** None (correctness module). The quiz focuses on the counting.

**Predict first.** Parameters and FLOPs per token for a Llama-3-8B-shaped config, using config values read from the model's published `config.json` (cited at build time).

**Sources.** kipply, *Transformer Inference Arithmetic* · nanoGPT `model.py` · HF `modeling_llama.py` · llm.c `train_gpt2.c` (CPU forward pass) · Silicon to Scale ch 2 (transformer deep dive), ch 1 (performance mindset) · handbook `llm-inference-basics/*`.

**Go down when…** a matmul is unclear: P0.4. **Go up to:** P1.2.

---

## P1.2 — Prefill vs decode, and the KV cache

**Objectives.**

1. Explain why prefill is compute-bound (a big GEMM) and decode is memory-bound (a GEMV per token at small batch).
2. Size the KV cache: `2 · n_layers · n_kv_heads · head_dim · bytes · tokens`, and per-sequence and per-batch totals.
3. Show how batching turns decode GEMVs back into GEMMs and moves decode up the roofline.
4. Implement a KV cache in the NumPy model and measure the speedup over recomputation.

**Examples.** `01_no_cache_vs_cache.py` (naive vs optimized) · `02_kv_sizer.py` · `03_arith_intensity_vs_batch.py` (plots AI vs batch for decode).

**Exercises.** (1) KV size function, tested against hand-computed cases · (2) the batch at which decode becomes compute-bound for a given GPU spec (pytest with fixture specs) · (3) cache correctness: cached logits = uncached logits · (4) *(hard)* sliding-window KV: memory vs quality trade-off on the tiny model.

**Bench.** `bench/cache_bench.py` (T0): `seq len | no-cache ms/token | cache ms/token | speedup`.

**Animation (required).** `animations/p1-prefill-decode.html`: tokens enter in one prefill wave, then decode one per step. The KV cache bar grows each step, and a TTFT/ITL timeline is drawn beneath.

**Sources.** handbook `how-does-llm-inference-work.md` (AutoregressiveDecodeStepper, ContextWindowSimulator) · Silicon to Scale ch 6 (prefill/decode regimes), ch 11 (KV cache) · kipply.

**Go up to:** P2.3, P6.3.

---

## P1.3 — Latency and throughput metrics

**Objectives.**

1. Define TTFT, ITL, TPOT, end-to-end latency, throughput (req/s, output tok/s) and goodput (throughput that meets the SLO).
2. Compute them from a per-token timestamp log, including percentiles (p50/p90/p99).
3. Explain the latency–throughput trade-off and why a mean hides tail latency.
4. Map each metric to the vLLM Prometheus metric that measures it (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_queue_time_seconds`, …; names read from `vllm/docs/usage/metrics.md` at the pinned SHA).

**Examples.** `01_metrics_from_log.py` (parses a JSONL of token timestamps) · `02_goodput.py` · `03_fake_server.py` (an asyncio OpenAI-compatible mock with configurable prefill/decode costs; T0, reused by P2.7 and #6).

**Exercises.** (1) TTFT/ITL/TPOT computation (pytest) · (2) percentile from histogram buckets, the way Prometheus does it (`histogram_quantile` semantics) · (3) goodput under a two-part SLO · (4) *(hard)* stream with SSE and measure client-side vs server-side TTFT.

**Bench.** `bench/mock_load.py` drives the mock server at increasing request rates and prints `rate | p50 TTFT | p90 TTFT | p50 ITL | goodput`.

**Sources.** handbook `llm-inference-metrics.md`, LatencyTimelineVisualizer · vLLM `docs/design/metrics.md` · Silicon to Scale ch 12 (benchmarking).

**Go up to:** P2.4, P3.5.

---

## P1.4 — GPU architecture and the roofline

**Objectives.**

1. Describe the GPU hierarchy: GPCs → SMs → warp schedulers → tensor cores; registers, shared memory/L1, L2, HBM.
2. Read a whitepaper and extract the dense peak FLOP/s per precision, HBM bandwidth, L2 size and SM count. Explain the difference between sparse and dense marketing numbers.
3. Build a GPU roofline for T4, A10G, L4, L40S, A100 and H100. These are the GPUs in §5's AWS table. Every number is cited to a whitepaper page.
4. Place decode GEMV, prefill GEMM and attention on it.

**Examples.** `01_gpu_specs.yaml` (each entry has `source:`; a loader refuses entries without one) · `02_gpu_roofline.py` · `03_measure_bw.py` (*optional T2*: a PyTorch device-to-device copy that measures achieved bandwidth vs the spec).

**Exercises.** (1) ridge point per GPU (pytest against the YAML) · (2) classify ops as memory- or compute-bound · (3) the "% of peak" helper used by every later bench script · (4) *(hard, T2)* measure achieved HBM bandwidth with your own copy kernel vs `torch.Tensor.copy_`.

**Predict first (worked prediction from the prompt).** Llama-3-8B fp16 on an L4, batch 1: weights ≈ 16 GB. Decode ceiling ≈ HBM bandwidth ÷ 16 GB. The L4 bandwidth figure is taken from the cited Ada/L4 document, and if the doc can't be verified the lesson prints a `TODO(verify)`. Measured in P2.1 with `TODO(run-on: g6.xlarge)` until run.

**Animation.** `animations/roofline.html` (GPU presets).

**Sources.** NVIDIA whitepapers (UNVERIFIED; required) · handbook `kernel-optimization/gpu-architecture-fundamentals/*` (SMFloorplan, WarpSchedulerVisualizer) and `choosing-the-right-gpu.md` (GPUTable) · Silicon to Scale ch 3a/3b, ch 4, Appendix A · Horace He, *Go Brrrr*.

**Go down when…** you want to know *why* the ceilings exist: P5.1–P5.2. **Go up to:** P1.5.

---

## P1.5 — Project: capacity calculator, D3

**Build.** A Python package and CLI, `s2s-capacity`, plus a static page in the site. Input: a model `config.json` (or HF id), a GPU from `gpu_specs.yaml`, dtype for weights and KV, TP degree, context length. Output: weights VRAM, activation/workspace estimate, KV bytes per token, max concurrent sequences at a given context, a decode tok/s ceiling per batch size, and a prefill time estimate.

**Objectives.** Pull P1.1–P1.4 into one tested tool that every later phase uses (P2 bakeoffs, P3 pod sizing, P4 TP choice).

**Exercises (the tests are the spec).** (1) weights VRAM within 2% of `safetensors` file sizes for 3 pinned models · (2) KV formula vs hand calc · (3) max-batch vs vLLM's reported KV-cache capacity, which is logged at startup (`TODO(run-on: g6.xlarge)` until measured) · (4) a cross-check table against the handbook's GPU memory calculator for 3 configs, with differences explained · (5) *(hard)* add TP and FP8 KV, and an MoE (active vs total params) mode.

**Bench.** None. The output table is the deliverable: `model | GPU | dtype | weights GB | KV GB/1k tok | max seqs @ ctx | decode ceiling tok/s @ batch 1/8/32`.

**Lives in.** `platform/tools/capacity/` (first code in the platform repo).

**Sources.** handbook `calculating-gpu-memory-for-llms.md` · Silicon to Scale ch 11 · kipply.
