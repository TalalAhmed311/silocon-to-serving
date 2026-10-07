# P6.5: Spec-decode verification and GPU sampling

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the NumPy references and the PyTorch sampler on CPU. ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for exercise 1's GPU statistics and exercise 3's benchmark |
| **Time** | ≈25 min reading + ≈8 h hands-on |
| **Prerequisites** | P2.6 (speculative decoding, `platform/specdec`), P5.4 (reductions, scans), Lane B L4 #60 top-p, L4 #104 min-p, L6 #87 verification |
| **You will build** | `sampling.py` (batched temperature / top-k / top-p / min-p without a full sort, seeded) and `spec_verify.py` (batched exact rejection sampling) in #0 v1 |

## Learning objectives

1. Apply temperature, top-k, min-p and top-p in the right order, per request, in one batched pass.
2. Find the top-p cut-off **without sorting the vocabulary**: binary search on a probability threshold.
3. Make sampling reproducible per request (seeded generators) while the batch composition changes every step.
4. Verify k draft tokens in one target pass with the rejection rule, batched, and prove it preserves the target distribution statistically.

---

## 1. The filters

For one row of logits `z` (vocabulary V):

```
p = softmax(z / T)                       T = 0 → argmax (greedy), no randomness
top-k:  keep p_i ≥ (k-th largest p)
min-p:  keep p_i ≥ min_p · max(p)        scales with the model's confidence
top-p:  keep the smallest set whose mass ≥ top_p   ⇔   keep p_i ≥ τ, τ = the largest threshold with mass(p ≥ τ) ≥ top_p
renormalise, sample
```

The top-p rewrite is the trick: mass(p ≥ τ) is **monotone decreasing in τ**, so τ can be found by bisection — 30–40 passes of "sum the probabilities above a threshold", each a parallel reduction, instead of a V-length sort per row. With V ≈ 128k and a batch of 256, that's the difference between a sort-bound and a bandwidth-bound sampler. `top_p_threshold_np` is the reference; `sample_torch` does the same batched on the GPU (one `[B, V]` comparison + row sum per iteration). Production samplers go further (FlashInfer's sorting-free rejection sampling; vLLM's `vllm/v1/sample/` at the pin — read which method it uses). Lane B L4 #60 and #104 are the kernel versions.

**Reproducibility.** Each request gets its own `np.random.Generator` (or `torch.Generator`) seeded from `seed`, consumed once per sampled token. Batch composition then can't change a request's random stream. (`test_seeded_sampling_reproducible`.)

## 2. Batched verification

From P2.6: the draft proposes x₁…x_k with probabilities q; one target pass gives p at k+1 positions. For each position in order: accept x_i with probability min(1, p_i(x_i)/q_i(x_i)); on the first rejection, sample from the residual `max(p_i − q_i, 0)` normalised, and stop; if all k are accepted, sample a bonus token from p_{k+1}. The emitted tokens are distributed **exactly** as the target alone would produce (Leviathan et al., Theorem 1).

`spec_verify.verify_batch` does this for B sequences at once. The GPU version (Lane B L6 #87) runs one thread block per sequence: a prefix scan over the accept flags finds the first rejection, then one residual sample.

> **Predict first.** k = 4, per-token acceptance α = 0.8. Expected tokens per target pass? (`(1 − α^{k+1}) / (1 − α)` ≈ 3.36, from `platform/specdec/core.py::expected_tokens_per_round`.) What does the verifier cost relative to the target forward pass? (Almost nothing: it reads B·(k+1)·V probabilities once.)

## 3. In the engine

`LLMEngine.step` samples with `sample_np` per row (simple, T0). The GPU path batches: gather the rows that produced a token into `[B, V]`, build per-row parameter tensors once per step, call `sample_torch`, copy B ints back. Integrating spec decoding into the scheduler means a sequence's decode chunk becomes `1 + k` tokens and `update()` may append up to `k + 1` tokens and must free any blocks allocated for rejected positions — exercise 4 in P6.7's stretch list.

## Walkthrough

```bash
uv run pytest platform/engine/v1/tests/test_sampling.py platform/engine/v1/tests/test_spec_verify.py -q
uv run pytest course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/exercises -q            # yours
uv run --extra torch pytest course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/exercises -m torch -q
uv run --extra torch python course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling/bench/sampler_bench.py --device cpu
```

## What you should see

- The threshold-bisection top-p keeps exactly the same set as the sort-based reference on 20 random Dirichlet distributions.
- The verification test: the first emitted token's empirical distribution matches p₀ within 0.03 over 4,000 trials.
- `sampler_bench.py` prints ms per batch for the sort-based and bisection samplers at V = 32k/128k. On CPU the numbers mean little; the GPU table is `TODO(run-on: g6.xlarge)`.

## Bench

| V | B | sort-based (ms) | bisection (ms) | GPU |
|---|---|---|---|---|
| 128k | 64 | `TODO(run-on: g6.xlarge)` | | L4 |
| 128k | 256 | | | L4 |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Distribution preservation of batched verification (statistical)](exercises/01-verify.md) | T0 → T2 | `test_my_sampling.py::test_verify_*` |
| 2 | [Top-p by threshold search vs a sort-based reference](exercises/02-top-p.md) | T0 (+ torch) | `test_my_sampling.py::test_top_p_*` |
| 3 | *(hard)* [Fused min-p + top-k sampler, benchmarked](exercises/03-fused.md) | T2 | `bench/sampler_bench.py` |

## Common mistakes

- Applying top-p before temperature: the kept set changes with T.
- Running the top-p threshold search on an **unnormalised** vector after top-k/min-p zeroed some entries: the mass test then compares against the wrong total. Renormalise first (min-p doesn't care: top-k never removes the maximum).
- One global RNG for the batch: a request's tokens then depend on which other requests happened to be in its batch.
- Verification that samples the residual from `p − q` without clamping at 0, or that forgets the bonus token when all k are accepted.

## Go deeper

- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML '23); Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023).
- Nguyen et al., *Min-p sampling* (arXiv 2407.01082). Holtzman et al., *The Curious Case of Neural Text Degeneration* (top-p).
- vLLM `vllm/v1/sample/sampler.py`, `vllm/v1/sample/ops/topk_topp_sampler.py`, `vllm/v1/sample/rejection_sampler.py` at `v0.31.0` (verify paths at the pin).

**Next:** [P6.6 CUDA graph capture](../P6.6-cuda-graph-capture/README.md).
