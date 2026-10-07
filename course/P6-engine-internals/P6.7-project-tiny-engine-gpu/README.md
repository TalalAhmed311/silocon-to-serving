# P6.7: Project — Tiny Inference Engine on the GPU (#0 v1)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the engine logic, the paged NumPy runner and the HTTP server (fake model). ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for the GPU runner, graphs, and the benchmark against vLLM |
| **Time** | ≈20 min reading + ≈14 h hands-on |
| **Prerequisites** | P6.1–P6.6, P5.10 (D4 kernels as PyTorch ops), P2.3 (#4 loadgen), P2.7 (#6 gateway) |
| **You will build** | **#0 v1** in `platform/engine/v1`: the P0 engine grown into a GPU serving engine, behind an OpenAI-compatible API, registered in the gateway, benchmarked against vLLM on the same GPU and model |

## What you ship

```
platform/engine/v1/
  s2s_engine/
    sequence.py        request state                          (P6.1)
    scheduler.py       continuous batching, chunked prefill,   (P6.2)
                       preemption, priorities + aging
    block_manager.py   paged KV, chained block hashes, CoW     (P6.3)
    radix_cache.py     token-level prefix cache                (P6.4, optional in the engine)
    sampling.py        temperature/top-k/top-p/min-p, seeded   (P6.5)
    spec_verify.py     batched rejection sampling              (P6.5)
    model_runner.py    FakeRunner (T0) · NumpyPagedRunner (T0) · TorchRunner (T2)
    cuda_graph.py      decode graphs per batch bucket          (P6.6)
    engine.py          LLMEngine: the step loop
    server.py          /v1/completions, /v1/chat/completions, /metrics (vLLM metric names)
  tests/               everything above, T0 except test_torch_runner.py (gpu)
```

The reference implementation is complete and tested at T0. Your project is to **own** it: rebuild the pieces you've written in P6.2–P6.6 into it (or replace ours with yours module by module — the tests are the contract), swap in D4 kernels, then measure.

## Build steps

1. **T0 green.** `uv run pytest platform/engine/v1/tests -q`. `test_numpy_engine.py` is the strongest test in the phase: the paged engine with chunking and preemption produces exactly v0's greedy tokens.
2. **GPU parity.** On a `g6.xlarge` (see [aws.md](aws.md)): `uv run --extra torch pytest platform/engine/v1/tests/test_torch_runner.py -m gpu`. fp32 greedy must match the NumPy reference token for token; for bf16, compare logits within a stated tolerance and report the first divergence step instead.
3. **D4 kernels.** Replace, one at a time, with a parity check after each: RMSNorm → `torch.ops.s2s.fused_add_rms_norm` (P5.10); the gather attention in `TorchRunner.decode` → your paged decode kernel (P5.8 ex. 4 / P6.3 ex. 4); the projections → your HGEMM or cuBLAS (`torch.matmul`) — keep cuBLAS where yours loses, and say so in the table.
4. **Serve.** `S2S_RUNNER=torch S2S_MODEL=<converted model dir> S2S_API_KEY=… uv run uvicorn s2s_engine.server:app --app-dir platform/engine/v1 --host 127.0.0.1 --port 8002`. Convert a model with `platform/engine/v0/tools/convert_hf_model.py` (a ≤ 1B Llama-architecture model fits an L4 comfortably in fp32 for parity, bf16 for the benchmark).
5. **Gateway.** Add a backend to your gateway config (P2.7):
   ```yaml
   - name: s2s-v1
     url: http://127.0.0.1:8002
     models: ["s2s-engine-v1"]
     api_key_env: S2S_API_KEY
   ```
6. **Benchmark against vLLM** (same GPU, model, dtype, max context, and as close as possible the same `max_num_seqs` / `max_num_batched_tokens`): `bench/compare.sh` runs #4 against both and prints the knee and the table below.
7. **Profile.** nsys for one decode step at B = 8 (eager and graphs), ncu for your top 3 kernels by time. Keep the reports out of git; put the summary table in your README.

## Acceptance tests

- [ ] `platform/engine/v1/tests` pass at T0 (CI runs them).
- [ ] Greedy outputs token-identical to the NumPy/PyTorch reference for 8 prompts in fp32 on the GPU, with graphs on and off; bf16 within your stated logit tolerance.
- [ ] Survives #4 at 2× its knee rate for 5 minutes with no errors (preemptions are fine; they must show in `/metrics`).
- [ ] #4 knee report vs vLLM, and the bench table filled in.
- [ ] ncu/nsys summaries for the top 3 kernels.

## Bench

| engine | batch | TTFT p50 (ms) | ITL p50 (ms) | tok/s | % of vLLM |
|---|---|---|---|---|---|
| vLLM `v0.31.0` | 1 | `TODO(run-on: g6.xlarge)` | | | 100 % |
| vLLM | 32 | | | | 100 % |
| #0 v1, eager, gather attention | 1 / 32 | | | | |
| #0 v1, graphs, gather attention | 1 / 32 | | | | |
| #0 v1, graphs, D4 paged attention + RMSNorm | 1 / 32 | | | | |

Expect to be well below vLLM at first (Python step loop, gather attention, no fused kernels); the point is to know **where** each gap comes from, using your profile. Don't quote a percentage you didn't measure.

## Stretch

1. **Async scheduling:** overlap `scheduler.step()` for step t+1 with the GPU work of step t (vLLM v1 does this). What state does the scheduler need before step t's tokens are known?
2. **Spec decode in the loop:** decode chunks of 1 + k tokens with a small draft model; `update()` appends up to k + 1 tokens and returns rejected positions' blocks (P6.5).
3. **Radix cache as the KV owner** instead of block hashing (P6.4 ex. 2), then compare hit rates on the #4 shared-prefix workload (`--shared-prefix 512 --shared-fraction 0.8`).
4. **Tensor parallel decode** on a 4-GPU box with your P4.2 sharding and NCCL all-reduces.

## Common mistakes

- Benchmarking against vLLM with different limits: vLLM's defaults may allow far larger batches; match `max_num_seqs`, `max_num_batched_tokens`, max model length, and dtype, and print both configs next to the result.
- Comparing your fp32 engine with vLLM in bf16.
- Leaving the server bound to `0.0.0.0` without an API key on a cloud box. Bind to `127.0.0.1`, forward with SSM.

## Go deeper

- nano-vllm (the whole thing, again — now you'll read it differently). vLLM v1 architecture blog post and `docs/design/` at the pin.
- Sarathi-Serve, Orca, PagedAttention, SGLang papers (cited in P6.2–P6.4).

**Next:** [Capstone](../../capstone/syllabus.md).
