# #0 v1 — the GPU serving engine

The P0 CPU engine (`../v0`) grown into a serving engine: continuous batching with chunked prefill, paged KV with prefix reuse, a radix prefix cache, batched sampling and speculative verification, CUDA-graph decode, and an OpenAI-compatible server. Built through [P6](../../../course/P6-engine-internals/syllabus.md); the project brief is [P6.7](../../../course/P6-engine-internals/P6.7-project-tiny-engine-gpu/README.md).

| file | what | tier | tests |
|---|---|---|---|
| `s2s_engine/sequence.py` | request state, `SamplingParams` | T0 | (all) |
| `s2s_engine/scheduler.py` | token budget, chunked prefill, preemption by recompute, priorities + aging | T0 | `test_scheduler.py` |
| `s2s_engine/block_manager.py` | block tables, ref counts, chained sha256 block hashes, cached-free LRU, CoW | T0 | `test_block_manager.py` |
| `s2s_engine/radix_cache.py` | token-level radix prefix cache + a naive trie for tests | T0 | `test_radix_cache.py` |
| `s2s_engine/sampling.py` | NumPy and batched PyTorch sampling; top-p by threshold bisection | T0 | `test_sampling.py` |
| `s2s_engine/spec_verify.py` | batched exact rejection sampling | T0 | `test_spec_verify.py` |
| `s2s_engine/model_runner.py` | `FakeRunner` (T0), `NumpyPagedRunner` (T0, real Llama), `TorchRunner` (T2) | T0/T2 | `test_numpy_engine.py`, `test_torch_runner.py` |
| `s2s_engine/cuda_graph.py` | decode graphs per padded batch bucket | T2 | `test_torch_runner.py` |
| `s2s_engine/engine.py` | `LLMEngine` step loop, request validation | T0 | `test_scheduler.py` |
| `s2s_engine/server.py` | `/v1/completions`, `/v1/chat/completions` (+SSE), `/v1/models`, `/health`, `/metrics` | T0 | `test_server.py` |

```bash
uv run pytest platform/engine/v1/tests -q                                       # T0: everything except the GPU runner
S2S_RUNNER=fake uv run uvicorn s2s_engine.server:app --app-dir platform/engine/v1 --port 8002
curl -s localhost:8002/v1/completions -H 'content-type: application/json' -d '{"prompt": "5 17 3", "max_tokens": 8}'
uv run --extra torch pytest platform/engine/v1/tests -m gpu                     # T2, TODO(run-on: g6.xlarge)
```

Design notes:

- **One counter drives everything:** `Sequence.num_computed` (tokens with K/V in the cache). Decode, prefill chunks, prefix hits and recompute are all "schedule the uncomputed tokens".
- **Liveness:** requests that can never fit are rejected at `add_request`; running work is never preempted by admission; a waiting request never holds blocks; so the oldest running request always eventually owns the whole pool.
- **The fake model is the scheduler's oracle:** logits are a hash of the token history, so any batching/chunking/preemption/caching bug changes outputs.
- **Graphs reserve block 0:** padded rows in a captured decode write there.
- **Not real tokenization:** the server takes token ids; other text is hashed one word → one id so #4 loadgen can drive it.

Status: T0 parts written and reviewed, `TODO(run)` — not yet executed in this repo; GPU parts `TODO(run-on: g6.xlarge)`. See [GAPS.md](../../../GAPS.md).
