# P1.2 — Prefill vs decode, and the KV cache

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) |
| **Time** | ≈30 min reading + ≈6 h hands-on |
| **Prerequisites** | P1.1 |
| **You will build** | a KV-cache sizer, an intensity-vs-batch model, and a cached vs uncached benchmark |

## Learning objectives

1. Explain why **prefill is compute-bound** (one big GEMM) and **decode is memory-bound** (a GEMV per token at small batch).
2. Size the KV cache: `2 · L · H_kv · h · bytes · tokens`, per sequence and per batch.
3. Show how batching turns decode GEMVs back into GEMMs and pushes decode up the roofline, and where that stops (the KV-cache reads).
4. Implement a KV cache in the NumPy model and measure its speedup over recomputation.

## Why this matters

Every serving-engine feature in P2 is a response to the facts in this module:

- continuous batching (raise decode's intensity)
- chunked prefill (stop compute-bound prefill from stalling memory-bound decodes)
- PagedAttention (stop wasting the KV memory that bounds batch size)
- prefix caching (skip prefill you have already done)
- FP8 KV (fit more sequences)
- disaggregation (run the two phases on different GPUs)

---

## 1. Two phases of one request

```
prompt (T tokens) ──▶ PREFILL: one forward pass over all T tokens at once ──▶ first token     (time = TTFT)
                     DECODE : one forward pass per new token, reading the cache ──▶ token 2, 3, … (time between = ITL)
```

**Prefill** multiplies a `[T, d]` activation matrix by each `[d, d']` weight matrix: a **GEMM**. Each weight byte is read once and used T times, so the intensity is ≈ T FLOP/byte (fp16). For T ≥ a few hundred, that is past the ridge of every GPU, so prefill is compute-bound. Its time grows ~linearly with T, and attention adds an O(T²) term.

**Decode** multiplies a `[B, d]` matrix (B = sequences in the batch) by each weight matrix. At B = 1 it is a **GEMV**: intensity ≈ 1 FLOP/byte (fp16), deep in memory-bound territory. Its time ≈ (weight bytes + KV bytes) / bandwidth, nearly independent of how many FLOPs you do.

## 2. The KV cache

Attention for the new token needs K and V of **every previous token**, at every layer. Recomputing them each step is O(t) extra GEMV work per token per layer. Storing them is the **KV cache**:

```
KV bytes per token = 2 (K and V) × L × H_kv × h × bytes_per_elem
```

For a 32-layer, 8-KV-head, h = 128 model in fp16: `2 · 32 · 8 · 128 · 2 = 131,072 B = 128 KiB per token`. At 8k context that is 1 GiB per sequence. **The KV cache, not the weights, is usually what limits how many sequences fit on the GPU.** P1.5's calculator and P2's engines revolve around this number.

> **Predict first.** A 24 GiB GPU holds a model with 16 GiB of fp16 weights. Leave ~1.5 GiB for activations and the CUDA context, so ~6.5 GiB remain for KV. With 128 KiB per token, how many tokens of KV fit? How many concurrent 2k-token conversations is that? And with an FP8 KV cache? (≈ 53k tokens → ≈ 26 sequences; FP8 doubles both.) Check yourself with `examples/02_kv_sizer.py`.

## 3. Batching moves decode up the roofline, until the KV cache stops it

With B sequences decoding together, each weight is read once per step and used B times: weight intensity ≈ B. But each sequence also reads **its own** KV cache, which is not shared, so those bytes grow with B:

```
bytes/step ≈ P·b_w  +  B · t · kv_bytes_per_token
FLOPs/step ≈ 2·P·B  +  B · 4·d·L·t
```

At small context the weights dominate and intensity ≈ B. At long context the KV term dominates and intensity tends to a constant (attention reads every cached byte to do ~1 FLOP/byte of work), however big the batch. `examples/03_arith_intensity_vs_batch.py` plots both regimes for a chosen GPU.

## 4. Cost of *not* caching

Without a cache, generating token t means re-running the forward pass over all t tokens: a growing prefill every step, O(n²) work for n tokens. `examples/01_no_cache_vs_cache.py` runs both versions on the tiny NumPy model and checks the logits match. `bench/cache_bench.py` times them: the uncached time per token grows linearly with position, while the cached time stays nearly flat.

---

## Walkthrough

```bash
D=course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache
uv run python $D/examples/01_no_cache_vs_cache.py
uv run python $D/examples/02_kv_sizer.py --layers 32 --kv-heads 8 --head-dim 128 --gpu-gb 24 --weights-gb 16
uv run python $D/examples/03_arith_intensity_vs_batch.py --gpu L4
uv run python $D/bench/cache_bench.py
```

## What you should see

- `01`: `cached == uncached logits (max |Δ| < 1e-4)` at every position.
- `02`: KV per token, per 1k/8k/32k-token sequence, and max concurrent sequences, all computed exactly from the inputs.
- `03`: a plot (`results/intensity_vs_batch.png`) where intensity rises ~linearly with B at short context and flattens at long context, with the GPU's ridge drawn in. The GPU numbers come from `P1.4/gpu_specs.yaml` and are marked UNVERIFIED until P1.4 exercise 1 is done.
- `cache_bench`: `seq len | no-cache ms/token | cache ms/token | speedup`. **TODO(run): paste yours.** Expected shape: the speedup grows roughly linearly with sequence length.

## Exercises

```bash
uv run pytest course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [KV size function](exercises/01-kv-size/README.md) | easy | hand-computed cases incl. GQA, FP8, multi-sequence |
| 2 | [When does decode become compute-bound?](exercises/02-decode-crossover/README.md) | medium | crossover batch for fixture GPUs at short and long context |
| 3 | [KV cache correctness](exercises/03-cache-correctness/README.md) | medium | your cached forward pass == full recompute at every position |
| 4 | [Sliding-window KV](exercises/04-sliding-window/README.md) | hard | memory bound + exactness inside the window; quality vs window size report |

## Common mistakes

- **Using `num_attention_heads` instead of `num_key_value_heads`** in the KV formula: 4–8× too big for GQA models.
- **Forgetting the factor 2** (K *and* V).
- **"Batching makes decode compute-bound."** Only at short context. At long context the per-sequence KV reads keep it memory-bound.
- **Counting prefill time as decode time.** TTFT includes the whole prefill; ITL doesn't.

## Go deeper

- Modular handbook, *How does LLM inference work*: the AutoregressiveDecodeStepper, ContextWindowSimulator and LatencyTimelineVisualizer interactives.
- Silicon to Scale ch. 6 (inference systems: the prefill/decode regime analysis) and ch. 11 (KV-cache engineering).
- kipply, *Transformer Inference Arithmetic*: the KV-cache and "flops vs memory" sections.

## Go down when…

You want to see why batched decode is still a GEMM → [P0.4 §2](../../P0-systems-primer/P0.4-simd-and-roofline/README.md). **Next:** [P1.3 Metrics](../P1.3-latency-and-throughput-metrics/README.md). **Comes back in:** P2.3 (batching), P6.3 (paged KV).

## Animation

[`animations/p1-prefill-decode.html`](../../../animations/p1-prefill-decode.html): prompt tokens enter in one prefill wave, then one token per decode step. The KV cache grows with each step, and the TTFT/ITL timeline is drawn underneath.
