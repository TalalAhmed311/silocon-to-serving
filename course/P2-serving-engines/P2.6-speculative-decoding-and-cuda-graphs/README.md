# P2.6 — Speculative decoding and CUDA graphs (+ #10)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the math, the exact sampling rule and the prototype on toy and tiny models; ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for real draft/target pairs and vLLM's spec decode ([aws.md](aws.md)) |
| **Time** | ≈40 min reading + ≈12 h hands-on (≈4 GPU-hours) |
| **Prerequisites** | P1.2, P2.1, P2.3 |
| **You will build** | **#10**: `platform/specdec`, exact speculative sampling with acceptance accounting, against vLLM's built-in implementation |

## Learning objectives

1. Derive the expected tokens per target pass from the acceptance rate α and the draft length k, and the wall-clock speedup given the draft's cost.
2. Compare draft-model, n-gram (prompt lookup) and head-based (Medusa, EAGLE) proposers.
3. Implement speculative sampling so it **provably preserves the target's distribution**, and test that property statistically.
4. Explain why CUDA graphs speed up small-batch decode, and what they constrain.

## Why this matters

At batch 1, decode is memory-bound: the GPU reads every weight to produce **one** token, and the tensor cores sit idle (P1.4). Speculative decoding spends that idle compute. A cheap draft proposes k tokens, and the target checks all of them in **one** pass: a (k+1)-token mini-prefill that costs about the same as one decode step. When the guesses are good, you get several tokens per weight read. It is the main latency lever left once batching is maxed out, and it trades throughput for latency, which matters for how you deploy it.

---

## 1. The rule (Leviathan et al. 2211.17192; Chen et al. 2302.01318)

For each proposed token `x_i`, with draft probability `q_i(x)` and target probability `p_i(x)`:

1. accept with probability `min(1, p_i(x_i) / q_i(x_i))`
2. on the **first** rejection, sample a replacement from `norm(max(0, p_i − q_i))`, discard the rest of the proposals and stop
3. if all k are accepted, sample one **bonus** token from `p_{k+1}`, which the target computed anyway

**Theorem:** the emitted sequence has exactly the distribution of sampling from p alone. The draft only affects *speed*. Exercise 1 checks this with a χ² test. A subtle bug, such as resampling from `p` instead of the residual, breaks the theorem, and the test catches it.

## 2. How much faster?

If each proposal is accepted independently with probability α, the expected number of tokens per target pass is:

```
E[tokens] = 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α)
```

If a draft step costs `c` target steps, the speedup is `E[tokens] / (k·c + 1)`.

| α | k | E[tokens] | speedup at c = 0.05 | at c = 0.2 |
|---|---|---|---|---|
| 0.6 | 4 | 2.31 | 1.92× | 1.28× |
| 0.8 | 4 | 3.36 | 2.80× | 1.87× |
| 0.8 | 8 | 4.33 | 3.09× | 1.67× |

(Computed by `examples/01_spec_math.py`; these are formula values, not measurements.) A higher k helps only while α is high **and** the draft is cheap. Real α depends heavily on the domain: code and structured text accept far more than open-ended chat (exercise 4).

## 3. Proposers

| Proposer | Draft cost | α | Notes |
|---|---|---|---|
| small draft model (same tokenizer) | a separate model, a few % of target FLOPs | medium | simplest; needs a compatible small model |
| **n-gram / prompt lookup** | ~free (string matching in the prompt) | high on copy-heavy tasks (code edits, RAG), low otherwise | no extra weights |
| **Medusa** heads | extra decoding heads on the target | medium | trained heads; tree verification |
| **EAGLE** (2401.15077) | a light autoregressive head over target features | high | trained; among the strongest published methods |

vLLM's implementations live in `vllm/v1/spec_decode/` at the pinned SHA, and the user docs are in `docs/features/speculative_decoding/`. The CLI takes a speculative config. Check `vllm serve --help` at `v0.31.0` for the current flag (`--speculative-config '{"method": "ngram", "num_speculative_tokens": 4, …}'` in recent versions: **UNVERIFIED**).

## 4. The throughput trade-off

Verification spends compute on proposals that get rejected. At high batch the GPU is already busy (decode moves toward compute-bound, P1.2 §3), so that wasted compute costs throughput. Spec decode helps **latency at low batch** and can *hurt* aggregate throughput at high batch. #10 measures both regimes.

## 5. CUDA graphs

A decode step at small batch is hundreds of tiny kernels. On the CPU side, each launch costs several µs. At batch 1 on a fast GPU, **launch overhead** can be a large fraction of the step. A **CUDA graph** records the whole sequence of kernels once and replays it with a single launch. The constraints:

- **static shapes and addresses**: capture one graph per batch-size bucket, and pad the batch up to the nearest bucket
- no host synchronization inside the graph
- graph memory is reserved per captured size

That is why vLLM spends start-up time and memory "capturing" graphs (P2.1 exercise 3), and why `--enforce-eager` exists. P6.6 builds a capture loop yourself. For the design, read `docs/design/cuda_graphs.md` and `vllm/compilation/cuda_graph.py` at the pinned SHA.

> **Predict first.** You measured eager vs graphs at batch 1 in P2.1. If kernel launches are ~5 µs each and a decode step launches ~600 kernels, what is the overhead per step? (~3 ms.) Compare that with your measured ITL difference. Then predict how it changes for a model twice as large (the overhead stays fixed while the step time doubles, so the relative gain halves).

---

## Walkthrough

```bash
D=course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs
uv run python $D/examples/01_spec_math.py                 # T0: the table above, plus the optimal k per (α, c)
uv run python $D/examples/02_spec_sampling_toy.py         # T0: exactness demo on a Markov "model"
uv run --extra torch python $D/examples/03_tiny_hf_spec.py  # T0: draft + target = two tiny random HF Llamas on CPU
# T2 (aws.md): real draft/target pair, vLLM spec decode, graphs on/off
```

## What you should see

- `01`: the table above, and the best k for each (α, c).
- `02`: the empirical next-token distribution with speculative sampling matches the target's within sampling noise (χ² p-value printed). A deliberately broken variant, resampling from `p` instead of the residual, does not match.
- `03`: acceptance rate and tokens per target call for two random tiny models (α is low, because random models disagree), plus a check that greedy speculative output equals greedy target-only output.
- T2 (`TODO(run-on: g6.xlarge)`): vLLM n-gram spec decode on a code-editing prompt set shows α well above chat, and an ITL improvement at batch 1 that shrinks or reverses at batch 32.

## Exercises

```bash
uv run pytest course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/exercises
```

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [The verification rule preserves the target distribution](exercises/01-exactness.md) | T0 | χ² on 2·10⁴ samples; a planted bug must fail |
| 2 | [E[tokens] formula vs simulation](exercises/02-formula.md) | T0 | simulated i.i.d. acceptance within 3% of the formula |
| 3 | [#10 with a KV cache, and vs vLLM](exercises/03-specdec-vs-vllm.md) | T0 test + T2 run | greedy equivalence on tiny models; your results table |
| 4 | [α by domain](exercises/04-alpha-by-domain.md) | T2 | written analysis + numbers |

## Common mistakes

- Resampling from `p` on rejection, instead of `max(0, p − q)` normalized.
- Comparing speedups at different batch sizes.
- Draft and target with different tokenizers: the proposals are then meaningless.
- Benchmarking spec decode with `temperature = 0` only. Greedy acceptance is just "argmax equal", which isn't representative of sampling workloads.

## Go deeper

- Leviathan et al. 2211.17192 · Medusa 2401.10774 · EAGLE 2401.15077.
- vLLM `vllm/v1/spec_decode/`, `docs/features/speculative_decoding/`, `docs/design/cuda_graphs.md`, `vllm/compilation/cuda_graph.py`.
- Modular handbook: *Speculative decoding*.
- Silicon to Scale ch. 13 (speculative decoding and MoE).

**Next:** [P2.7 Gateway basics](../P2.7-gateway-basics/README.md). **Comes back in:** P6.5 (verification on the GPU), P6.6 (graph capture).

## Animation

[`animations/p2-speculative-decoding.html`](../../../animations/p2-speculative-decoding.html): the draft proposes k tokens, the target verifies them in one pass, and tokens are accepted or rejected, with the residual resample and the bonus token.
