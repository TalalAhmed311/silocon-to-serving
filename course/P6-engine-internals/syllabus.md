# P6 — Engine internals

**Weeks 35–40 · Lane A ≈ 66 h · Tier T0 for scheduler/allocator/radix-tree logic (pure Python, fully tested in CI) + T2 for the GPU engine · Lane B: L6 daily**

**Thread through the phase:** read nano-vllm line by line, re-implement each subsystem in your own engine, map every concept to the production code in vLLM `vllm/v1/` and SGLang `python/sglang/srt/`, then ship **#0 v1**. That is the P0 engine ported to CUDA with continuous batching, paged KV (D2 grown up), your D4 kernels, plugged into #6 and benchmarked against vLLM with #4.

Pinned: nano-vllm `bb823b3e`, vLLM `v0.31.0`, SGLang `v0.5.21`.

| Module | Time | Project | Animations |
|---|---|---|---|
| P6.1 Reading nano-vllm | 0.5 h + 6 h | — | — (Mermaid call graph) |
| P6.2 Scheduler | 0.5 h + 10 h | — | **scheduler loop: waiting / running / preempted queues per step** |
| P6.3 Paged KV block manager | 0.5 h + 10 h | — | reuses **PagedAttention** (P2.3) |
| P6.4 Prefix cache (radix tree) | 0.5 h + 9 h | — | **radix-tree prefix cache insert / match / evict** |
| P6.5 Spec-decode verification and GPU sampling | 0.4 h + 8 h | — | reuses **speculative decoding** (P2.6) |
| P6.6 CUDA graph capture | 0.4 h + 7 h | — | — |
| P6.7 Project: Tiny Inference Engine (GPU) | 0.3 h + 14 h | **#0 v1** | — |

---

## P6.1 — Reading nano-vllm

**Objectives.** Trace one request through `nanovllm/engine/llm_engine.py` → `scheduler.py` → `block_manager.py` → `model_runner.py`, with `sequence.py` as the data model. Produce an annotated map: each function → its vLLM v1 counterpart (`vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/worker/gpu_model_runner.py`) and its SGLang counterpart (`python/sglang/srt/managers/scheduler.py`, `mem_cache/radix_cache.py`).

**Exercises.** (1) Concept-mapping table (reviewed against the quiz) · (2) instrument nano-vllm with timers and explain where time goes per step (T2) · (3) *(hard)* find one behavioral difference between nano-vllm and vLLM's scheduler and reproduce it with a trace.

**Sources.** nano-vllm (MIT; ≤15-line annotated excerpts only) · vLLM / SGLang (Apache-2.0).

## P6.2 — Scheduler

**Objectives.** (1) The step loop: admit from waiting, budget tokens (`max_num_batched_tokens`), mix prefill chunks with decodes, and preempt (recompute vs swap) when blocks run out. (2) Fairness, priorities and starvation. (3) Implement your scheduler as a pure-Python module with a fake model, so it is fully T0-testable.

**Exercises.** (1) FCFS continuous batching (property tests: every admitted request finishes, and no step exceeds its token budget) · (2) chunked prefill · (3) preemption by recompute with victim selection · (4) *(hard)* priority classes with an anti-starvation bound, proven by test.

**Animation (required).** `animations/p6-scheduler-loop.html`.

**Sources.** nano-vllm `scheduler.py` · vLLM `vllm/v1/core/sched/` · Sarathi-Serve · Orca.

## P6.3 — Paged KV block manager

**Objectives.** Grow D2 into a paged KV manager: block tables per sequence, append/allocate on boundaries, ref-counted sharing, copy-on-write on fork (beam/parallel sampling), and hashing of full blocks for prefix reuse. Then the GPU side: a paged decode-attention kernel that reads through the block table (from P5.8 exercise 4).

**Exercises.** (1) Block table invariants (property tests, T0) · (2) CoW on fork · (3) block-hash prefix reuse vs vLLM's hashing scheme (read `kv_cache_manager.py` / block pool code) · (4) paged decode attention kernel vs contiguous reference (T2) · (5) *(hard)* fragmentation and utilization stats under the #4 trace.

**Sources.** PagedAttention paper · nano-vllm `block_manager.py` · vLLM `vllm/v1/core/` · handbook `pagedattention.md`.

## P6.4 — Prefix cache (radix tree)

**Objectives.** Implement a radix tree over token IDs with insert, longest-prefix match, ref counts and LRU eviction of unreferenced leaves, following RadixAttention. Compare it with block-hash prefix caching (P6.3): which workloads favor which.

**Exercises.** (1) Insert/match/split correctness (property tests vs a naive trie) · (2) LRU eviction respecting ref counts · (3) hit rate on multi-turn chat and few-shot traces · (4) *(hard)* cache-aware scheduling (longest-prefix-first) and its effect on throughput.

**Animation (required).** `animations/p6-radix-tree.html`.

**Sources.** SGLang paper (2312.07104) · SGLang `python/sglang/srt/mem_cache/radix_cache.py` · handbook `prefix-caching.md`.

## P6.5 — Spec-decode verification and GPU sampling

**Objectives.** (1) Batched verification of k draft tokens in one target forward pass, with the rejection-sampling rule on GPU. (2) GPU sampling: temperature, top-k, top-p and min-p without a full sort (threshold search / radix select), with seeded reproducibility. (3) Integrate both into the engine.

**Exercises.** (1) Distribution-preservation test for GPU verification (T2, statistical) · (2) top-p kernel vs `torch.sort`-based reference · (3) *(hard)* min-p + top-k fused sampler, benchmarked.

**Sources.** Leviathan et al. · vLLM `vllm/v1/spec_decode/`, `vllm/v1/sample/` (verified at the pinned SHA) · LeetGPU *Speculative Decoding Verification*, *Top-p Sampling*, *Min-P Sampling* (L4/L6).

## P6.6 — CUDA graph capture

**Objectives.** (1) What a graph captures and what it forbids: static addresses, no host sync, fixed shapes. (2) Capture per batch-size bucket with padding, and memory pools. (3) Measure launch-overhead savings at small batch. (4) Read vLLM `docs/design/cuda_graphs.md` and `vllm/compilation/cuda_graph.py`.

**Exercises.** (1) Capture/replay a decode step in PyTorch (`torch.cuda.CUDAGraph`) · (2) batch-size buckets + padding in your engine · (3) *(hard)* graph-safe sampling.

**Sources.** CUDA Programming Guide (CUDA Graphs) · vLLM CUDA-graph design doc and capture code.

## P6.7 — Project: Tiny Inference Engine (GPU), #0 v1

**Build.** Port #0 v0 to CUDA C++ (host orchestration in C++ or Python + your kernels): paged KV (P6.3), scheduler (P6.2), radix prefix cache (P6.4, optional), sampling (P6.5), graphs (P6.6), and D4 kernels (RMSNorm, RoPE, GEMM/HGEMM or cuBLAS fallback, decode attention). Expose an OpenAI-compatible endpoint, register it as a backend in #6, and benchmark it against vLLM with #4 on the same GPU and model.

**Acceptance tests.** Greedy outputs token-identical to the P0.5/PyTorch reference for N prompts (fp32) or within a stated logit tolerance (fp16) · #4 knee report vs vLLM · ncu/nsys reports for the top 3 kernels.

**Bench.** `engine | batch | TTFT p50 | ITL p50 | tok/s | % of vLLM`.

**Lives in.** `platform/engine/` (#0 v0 and v1 share the repo; v0 stays the CPU reference).
