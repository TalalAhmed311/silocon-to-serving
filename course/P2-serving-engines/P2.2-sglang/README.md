# P2.2 — SGLang

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange), the same instance as P2.1 (`g6.xlarge`, [aws.md](aws.md)). Use a separate venv from vLLM |
| **Time** | ≈20 min reading + ≈6 h hands-on (≈3 GPU-hours) |
| **Prerequisites** | P2.1 |
| **Pinned** | SGLang `v0.5.21` (`sgl-project/sglang@a34cba3c`), from `env/requirements-sglang.txt` |

## Learning objectives

1. Serve the same model with SGLang (`python -m sglang.launch_server`) and query it with the **same** OpenAI client.
2. Map SGLang's flags to vLLM's, concept by concept.
3. Run the same benchmark on both and put the results in **one** table.
4. Know at a high level where SGLang forms a batch (`python/sglang/srt/managers/scheduler.py`) and why its **RadixAttention** prefix cache wins on multi-turn and few-shot workloads.

## Why this matters

You'll be asked "vLLM or SGLang?" in every infra role. The honest answer is "it depends on the workload, and here's my measured table". Both are continuous-batching engines with paged KV caches. They differ in the prefix cache design (block hashing vs a radix tree, which P6.4 builds), in scheduling policy defaults, in the kernel backends, and in the frontend: SGLang ships a structured-generation DSL.

---

## 1. Flag mapping

Verify each row with `--help` at your pinned versions. The names below are the author's understanding and are **UNVERIFIED** until exercise 1 checks them automatically.

| Concept | vLLM (`vllm serve`) | SGLang (`sglang.launch_server`) |
|---|---|---|
| model path | positional `MODEL` | `--model-path` |
| bind / port | `--host` `--port` | `--host` `--port` |
| auth | `--api-key` | `--api-key` |
| fraction of VRAM for weights + KV | `--gpu-memory-utilization` | `--mem-fraction-static` |
| context length | `--max-model-len` | `--context-length` |
| max running sequences | `--max-num-seqs` | `--max-running-requests` |
| prefill chunk size | `--max-num-batched-tokens` | `--chunked-prefill-size` |
| prefix cache | `--enable-prefix-caching` | on by default (radix cache); `--disable-radix-cache` turns it off |
| tensor parallel | `--tensor-parallel-size` | `--tp-size` |
| KV dtype | `--kv-cache-dtype` | `--kv-cache-dtype` |
| no CUDA graphs | `--enforce-eager` | `--disable-cuda-graph` |

## 2. Where the batch is formed

Open `python/sglang/srt/managers/scheduler.py` at the pinned SHA and find the event loop. Each iteration it:

1. receives new requests
2. tries to build a prefill batch from the waiting queue: match each request's prefix in the radix cache, then admit it while token budgets allow
3. otherwise runs a decode step for the running batch
4. processes the outputs

vLLM's V1 scheduler (`vllm/v1/core/sched/scheduler.py`) does the same job with a unified token budget per step. You'll read both line by line in P6. For now, note the **policy knob** SGLang exposes (`--schedule-policy`, e.g. longest-prefix-match first): with a radix cache, it reorders the queue to maximize cache hits.

## 3. The workload where the radix cache wins

Multi-turn chat: turn *k* of a conversation sends the whole history again. A prefix cache turns all of that prompt except the newest message into a cache hit, so prefill shrinks to the new tokens only, and so does TTFT. `examples/02_multiturn.py` replays N conversations of M turns each against either server and reports TTFT per turn. With a working prefix cache, TTFT stays flat as the history grows. Without one, it grows with the history.

> **Predict first.** A 2,000-token system prompt shared by every request, and 50-token user messages. With a perfect prefix cache, what fraction of prefill work is saved? (2000 / 2050 ≈ 97.6%.) What happens to TTFT? Measure the hit rate from the server metrics (vLLM: prefix-cache hit counters in `/metrics`; SGLang logs cache hit info) and compare.

---

## Walkthrough (on the instance)

```bash
uv venv .venv-sglang && uv pip install -p .venv-sglang -r env/requirements-sglang.txt
source .venv-sglang/bin/activate
bash course/P2-serving-engines/P2.2-sglang/examples/01_serve.sh /opt/models/<model>
python course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/02_client.py --url http://127.0.0.1:30000 --api-key "$S2S_API_KEY"
python course/P2-serving-engines/P2.2-sglang/examples/02_multiturn.py --url http://127.0.0.1:30000 --api-key "$S2S_API_KEY"
# then the same against vLLM (P2.1's 01_serve.sh) and compare
```

## What you should see

`TODO(run-on: g6.xlarge)`. Expected shape:

- both servers answer the same client
- on the multi-turn replay, TTFT per turn stays nearly flat on both engines when prefix caching is on, and grows ~linearly with history when you disable it
- on a single-turn random-prompt workload, the two engines land within tens of % of each other

## Exercises

| # | Exercise | Tier | Check |
|---|---|---|---|
| 1 | [Flag-mapping table, verified by `--help`](exercises/01-flag-map.md) | T2 (T0 test on captured help text) | `test_flag_map.py` |
| 2 | [Multi-turn replay: measure the prefix-cache win](exercises/02-multiturn.md) | T2 (T0 test vs mock) | the replay's TTFT accounting on the mock |
| 3 | [Find a workload where vLLM wins, and explain it](exercises/03-vllm-wins.md) | T2 | written analysis with the table |

## Common mistakes

- Installing SGLang and vLLM into one venv. Their torch/flashinfer pins collide.
- Comparing engines at different `mem-fraction` / `gpu-memory-utilization` settings, which means different KV capacities.
- Declaring a winner from one workload.

## Go deeper

- SGLang paper (RadixAttention), arXiv 2312.07104.
- SGLang docs (`docs/` at the pinned SHA): server arguments, the frontend language.
- `python/sglang/srt/mem_cache/radix_cache.py`: you'll rebuild it in P6.4.

**Next:** [P2.3 Batching and caching](../P2.3-batching-and-caching/README.md).
