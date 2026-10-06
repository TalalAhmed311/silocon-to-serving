# P2.4 — Benchmarking methodology

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the methodology exercises (against `mockllm`); ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for GuideLLM and `vllm bench serve` ([aws.md](aws.md)) |
| **Time** | ≈25 min reading + ≈6 h hands-on (≈2 GPU-hours) |
| **Prerequisites** | P1.3, P2.3 |
| **Pinned** | GuideLLM `v0.8.0` (`vllm-project/guidellm@246f3b0f`), vLLM `v0.31.0` |

## Learning objectives

1. Distinguish closed-loop from open-loop load, and recognize **coordinated omission**.
2. Warm up, repeat, report the median and p90 with sample counts, and state the variance.
3. Run GuideLLM and `vllm bench serve`, and reconcile both with #4.
4. Write a benchmark report that a stranger can reproduce.

## Why this matters

Most LLM benchmark numbers you'll see are not reproducible: missing hardware, missing versions, missing flags, the wrong load model. Your portfolio teardown (#15) is only as credible as its method.

---

## 1. Closed loop, open loop, and coordinated omission

- **Closed loop:** N virtual users, each sends a request, waits for the response, then sends the next. Offered load *adapts to the server*: when the server slows down, the users send less. The queue never grows past N, so p99 latency looks great exactly when the system is overloaded.
- **Open loop:** requests arrive on a schedule, Poisson at rate λ, **regardless** of responses, the way real independent users do. Past capacity, the queue grows without bound and latency shows it.

**Coordinated omission** (Gil Tene's term) is the closed-loop measurement error: the load generator "coordinates" with the server and silently omits the requests that *would have been sent* while it was waiting. Exercise 1 shows it on the mock: a closed-loop client reports a flat p99 while an open-loop client at the same average rate shows the queue.

## 2. Repetition and variance

- **Warm up.** The first requests pay for CUDA-graph capture, torch.compile caches, the page cache for weights and allocator growth. Discard a warm-up phase.
- **Repeat the whole run** at least 3 times, and report the median of the run-level statistic plus its spread (min–max, or an interquartile range). One run is an anecdote.
- **Report counts.** "p99 TTFT over 120 requests" is barely more than the 2nd-largest sample. Say n.
- **Fix the seed** for workload generation, so that two engines see the same requests (`loadgen`'s `--seed`).
- **Change one thing at a time.**

## 3. The tools

| Tool | Load model | Good for | Notes |
|---|---|---|---|
| `platform/loadgen` (#4) | open loop, Poisson, seeded | knee finding, your own SLOs | yours: you know every line |
| GuideLLM | sweeps: synchronous, throughput, constant/Poisson rates | quick standardized sweeps against any OpenAI server | `src/guidellm/` at the pinned SHA |
| `vllm bench serve` | dataset-driven, request rate or `inf` | comparing vLLM configs, results comparable with vLLM's CI | `vllm/benchmarks/` and `docs/benchmarking/cli.md` |
| `vllm bench latency` / `throughput` | offline, no HTTP | engine-only numbers | not comparable with HTTP serving numbers |

**Reconcile before you trust.** Run all three at the same offered rate and workload shape. If they disagree by more than ~10%, find out why before you publish anything. Common culprits: tokens counted differently (words vs tokenizer tokens), TTFT from send vs from connect, `ignore_eos`, warm-up.

## 4. The report

`examples/report.py` generates a report skeleton from a loadgen JSON plus `--hardware/--versions/--flags`, and **refuses to render** if a required field is missing. Required fields:

- hardware (GPU, count, driver, CUDA), instance type and region
- engine and version (SHA), model repo and revision, dtype, quantization
- every non-default engine flag
- load generator and version, load model (open or closed), rate(s), duration, warm-up, seed, prompt/output length distribution and tokenizer
- statistics: n per point, median and p90 (p99 if n ≥ 1000), number of repeats and spread
- the raw JSON, linked

> **Predict first.** On the mock, closed loop with 32 users vs open loop at the *same* mean rate the closed loop achieved: which reports higher p99 TTFT, and by how much? Write a guess, then run exercise 1.

---

## Walkthrough

```bash
D=course/P2-serving-engines/P2.4-benchmarking-methodology
uv run uvicorn mockllm.server:app --app-dir platform --port 8001 &
uv run python $D/examples/01_closed_vs_open.py --url http://127.0.0.1:8001
PYTHONPATH=platform uv run python -m loadgen.cli --url http://127.0.0.1:8001 --rates 10 20 30 --duration 20 --out results/lg.json
uv run python $D/examples/report.py results/lg.json --hardware "laptop (mock)" --versions "mockllm@repo" --flags "defaults" > results/report.md
```

On the GPU instance, see [aws.md](aws.md) for the GuideLLM and `vllm bench serve` commands.

## What you should see

- `01_closed_vs_open`: closed loop at concurrency 32 reports a moderate, stable p99, while open loop at the same mean rate reports a much larger p99, because the queue builds during bursts. Exact numbers depend on the mock's settings. **TODO(run): paste them.**
- `report.py`: a Markdown report, or an error that lists the missing fields.

## Exercises

| # | Exercise | Tier | Check |
|---|---|---|---|
| 1 | [Show coordinated omission](exercises/01-coordinated-omission.md) | T0 | `test_co.py`: open-loop p99 > closed-loop p99 on an overloaded mock |
| 2 | [The report generator](exercises/02-report.md) | T0 | `test_report.py`: required fields enforced |
| 3 | [GuideLLM vs #4 vs `vllm bench serve`](exercises/03-reconcile.md) | T2 | written reconciliation table |

## Common mistakes

- Reporting `max_concurrency` results as if they were a request rate.
- Comparing an offline `vllm bench throughput` number with a serving number.
- Counting tokens with different tokenizers across tools.
- Publishing without the raw data.

## Go deeper

- vLLM `docs/benchmarking/` (`cli.md`, `sweeps.md`, `dashboard.md`) at the pinned SHA.
- GuideLLM README and `src/guidellm/benchmark/` at the pinned SHA.
- Modular handbook: *LLM performance benchmarks*.
- Silicon to Scale ch. 12 (benchmarking).
- Gil Tene's "How NOT to Measure Latency" talk (search for it), for the coordinated-omission argument.

**Next:** [P2.5 Quantization](../P2.5-quantization/README.md).
