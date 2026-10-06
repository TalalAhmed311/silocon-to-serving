# P2.3 — Batching and caching (+ #4 load test)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the simulator and #4 against `mockllm`; ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for the vLLM runs ([aws.md](aws.md)) |
| **Time** | ≈40 min reading + ≈12 h hands-on (≈4 GPU-hours) |
| **Prerequisites** | P1.2, P1.3, P2.1 |
| **You will build** | a discrete-event batching simulator, and **#4**: `platform/loadgen`, a load generator that finds the latency knee |

## Learning objectives

1. Explain **static**, **dynamic** and **continuous** (iteration-level) batching, and why the last one won.
2. Explain **chunked prefill** and why it protects ITL.
3. Explain **PagedAttention** blocks and **prefix caching** (block hashing).
4. Build #4: drive a server to KV-cache saturation and locate the latency knee.

---

## 1. Batching policies

Decode is memory-bound, so a batch of B sequences costs about the same per step as one sequence (P1.2 §3). Batching is the single biggest throughput lever. *How* you batch decides the latency.

| Policy | How | Problem |
|---|---|---|
| **static** | collect B requests, run them together until **all** finish | short requests wait for the longest one; new arrivals wait for the whole batch |
| **dynamic** | like static, but dispatch after a timeout even if the batch isn't full | still request-level: finished slots stay idle until the batch ends |
| **continuous** (Orca's iteration-level scheduling) | re-form the batch **every step**: finished sequences leave, waiting ones join | none of the above; this is what vLLM, SGLang and TensorRT-LLM do |

`examples/01_batching_sim.py` is a discrete-event simulator, with the same cost model shape as `mockllm`. It plots both policies' timelines for the same arrival trace. The waste in static batching is easy to see as the empty slots after short requests finish.

## 2. Chunked prefill

A new request's prefill of 4,000 prompt tokens is a big compute-bound step. If the scheduler runs it as one step, **every running sequence's next token waits for it**, and ITL spikes for everyone. Chunked prefill (SARATHI, then Sarathi-Serve) splits the prefill into chunks of `max_num_batched_tokens` and **piggybacks** decodes on each chunk:

```
step k:   [ 512 prefill tokens of the new request | 1 decode token × 63 running requests ]
```

Each step stays bounded in cost, so ITL stays bounded. The price is a slightly longer TTFT for the long prompt. In vLLM V1 the scheduler's per-step token budget *is* the chunk size.

## 3. PagedAttention and prefix caching

Reserving `max_model_len` of contiguous KV for every request wastes most of it, because most requests are short. **PagedAttention** stores KV in fixed-size **blocks**, for example 16 tokens each, allocated on demand. Each sequence has a **block table** mapping logical block i to physical block j, and attention kernels gather through it. Fragmentation drops to under one block per sequence. This is your D2 allocator (P0.3) on a GPU.

**Prefix caching:** if two requests start with the same tokens, their first blocks hold identical K/V. vLLM hashes each *full* block, chained with the hash of everything before it, and looks the hash up before allocating. On a hit, it maps the existing physical block (ref count + 1) and **skips that prefill**. Blocks are freed by ref count and evicted LRU when space runs out. Copy-on-write handles divergence.

## 4. #4: finding the knee

As offered load rises:

- the running batch grows until `max_num_seqs` or **KV capacity** caps it (`vllm:kv_cache_usage_perc` → 1)
- then requests queue (`vllm:num_requests_waiting` grows) and TTFT explodes
- under KV pressure vLLM **preempts** (`vllm:num_preemptions`): it evicts a running sequence's KV and recomputes it later

The **knee** is the last load at which the SLO still holds. `platform/loadgen` sweeps the rates and reports the knee, using the definition in `loadgen/analysis.py`.

> **Predict first.** On the mock with defaults (64 max seqs, base step 8 ms + 0.25 ms per running seq, 128-token mean outputs), a saturated batch has a step of 8 + 16 = 24 ms, so 64 / 0.024 ≈ 2,670 tok/s, so ≈ 2670 / 128 ≈ **21 req/s**. Find the knee with `loadgen.cli --rates 5 10 15 20 25 30`. Then predict the knee for vLLM on your GPU from your P2.1 batch-8 measurement and D3's max sequences, and check it.

---

## Walkthrough

```bash
D=course/P2-serving-engines/P2.3-batching-and-caching
uv run python $D/examples/01_batching_sim.py                         # T0: static vs continuous timelines -> results/batching.png
uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &
PYTHONPATH=platform uv run python -m loadgen.cli --url http://127.0.0.1:8001 --rates 5 10 15 20 25 30 --duration 30
PYTHONPATH=platform uv run python -m loadgen.plot results/loadgen.json
# T2: the same against vLLM (aws.md), with --enable-prefix-caching on/off and a shared-prefix workload:
PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 --shared-prefix 1000 --shared-fraction 0.8
```

## What you should see

- The simulator: continuous batching's mean and p90 latency are well below static's on the same trace. The output is deterministic (seeded) and the script prints both.
- The mock: goodput tracks offered load up to ≈ 20 req/s, then flattens while p90 TTFT climbs steeply. KV usage stays below 1 with the default mock capacity.
- vLLM (`TODO(run-on: g6.xlarge)`): the knee where `vllm:kv_cache_usage_perc` approaches 1 and `num_requests_waiting` starts growing. With the shared-prefix workload, prefix caching on moves the knee right and cuts TTFT.

## Exercises

```bash
uv run pytest course/P2-serving-engines/P2.3-batching-and-caching/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Continuous batching in the simulator](exercises/01-continuous-batching.md) | medium | beats static on mean latency for fixed traces; never exceeds `max_batch` |
| 2 | [Chunked prefill in the simulator](exercises/02-chunked-prefill.md) | medium | max ITL bounded by the chunk-step cost; TTFT cost reported |
| 3 | [#4 on the mock](exercises/03-loadgen-mock.md) | medium | the knee found within ±25% of the closed-form prediction |
| 4 | [Prefix caching on vLLM](exercises/04-prefix-caching.md) | hard (T2) | knee and TTFT with caching on vs off; `check_prefix.py` validates your recorded JSON |

## Common mistakes

- **Closed-loop load** ("N users, each sends the next request when the last returns"). It can never overload the server, so it never finds the knee. #4 is open-loop.
- **Too-short runs.** Queues need time to build: use ≥ 60 s per rate on a GPU.
- **Ignoring preemptions.** A knee caused by KV thrashing looks like a compute limit unless you read `vllm:num_preemptions`.
- **Prefix caching "doesn't work".** Usually the shared prefix is shorter than one block, or differs by one token (a timestamp in the system prompt).

## Go deeper

- Orca (OSDI '22, iteration-level scheduling) · SARATHI (2308.16369) and Sarathi-Serve (2403.02310) · PagedAttention (2309.06180).
- Modular handbook: *Static, dynamic and continuous batching* (BatchingSimulator, ChunkedPrefillVisualizer), *PagedAttention*, *Prefix caching*.
- vLLM `vllm/v1/core/sched/scheduler.py` and `vllm/v1/core/kv_cache_manager.py`: read-only for now. P6.2–P6.3 rebuild them.

## Go down when…

The knee is where you didn't expect it → P6.2 (scheduler) and P6.3 (block manager). **Next:** [P2.4 Benchmarking methodology](../P2.4-benchmarking-methodology/README.md).

## Animations

- [`animations/p2-continuous-batching.html`](../../../animations/p2-continuous-batching.html): static vs continuous batching on one trace, and chunked prefill letting decodes continue.
- [`animations/p2-paged-attention.html`](../../../animations/p2-paged-attention.html): logical → physical blocks, prefix sharing on a hit, copy-on-write.
