# P0.5 — Project: Tiny Inference Engine (CPU), #0 v0

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) |
| **Time** | ≈25 min reading + ≈6 h hands-on (plus as long as you like on exercise 5) |
| **Prerequisites** | P0.1–P0.4 (every one of them is used here) |
| **You will build** | the core ops of a Llama-architecture engine, tested against a NumPy reference, wired into `platform/engine/v0` |

## Learning objectives

1. Implement a Llama forward pass in C++: embedding, RMSNorm, RoPE, grouped-query attention with a KV cache, a SwiGLU MLP and the LM head.
2. Load weights zero-copy through your P0.2 mmap loader.
3. Use your P0.4 SIMD matvec and your P0.3 thread pool for every matrix multiply.
4. Implement temperature + top-p sampling with a seeded RNG.
5. Measure decode tokens/s and compare it with a **bandwidth-bound prediction**.

## Why this matters

Everything in P1–P6 is a refinement of this one loop: read every weight once, do a little math per weight, emit one token. When you later read vLLM's model runner or write a CUDA decode-attention kernel, this file is your mental model. And because you wrote the slow, clear version, you'll know exactly what the fast version is optimizing.

---

## 1. The model in one picture

```
token id ──▶ embed[vocab, d] ──▶ x (d floats)
for each of L layers:
    h = RMSNorm(x)                                    (per-channel weight)
    q = Wq·h   k = Wk·h   v = Wv·h                    (k, v have n_kv_heads ≤ n_heads: GQA)
    q, k = RoPE(q, pos), RoPE(k, pos)                 (rotate pairs by a position-dependent angle)
    append k, v to this layer's KV cache at pos
    a = softmax(q·Kᵀ/√hd) · V   per head              (attends to positions 0..pos only: causal)
    x = x + Wo·a                                      (residual)
    h = RMSNorm(x)
    x = x + Wdown·( SiLU(Wgate·h) ⊙ (Wup·h) )        (SwiGLU MLP, residual)
x = RMSNorm(x);  logits = Whead·x;  next token = sample(logits)
```

`platform/engine/v0/src/engine.hpp` is this picture in ~120 lines of C++. Read it now, top to bottom.

### RMSNorm

`y = x / sqrt(mean(x²) + ε) ⊙ w`. Unlike LayerNorm it doesn't subtract the mean. It is cheaper and works as well. Accumulate `Σx²` in double: for d = 4096, a float sum loses about 3 digits.

### RoPE (rotary position embedding)

RoPE encodes position by **rotating** pairs of q/k dimensions by an angle `pos × θ^(−2i/hd)`. The dot product `q·k` then depends only on the *difference* of positions (P1.1 exercise 4 proves it numerically). Two conventions exist:

- **interleaved:** pairs `(0,1), (2,3), …`. Used by the original paper and llama2.c.
- **rotate_half:** pairs `(i, i + hd/2)`. Used by Hugging Face Llama weights.

The engine uses **rotate_half**, because it loads HF weights. If your outputs are garbage after layer 0, check this first.

### Attention with a KV cache

Without a cache, step t recomputes K and V for all t previous tokens: O(t²) work in total per token. With a cache, each step computes K and V for **one** new token, appends them, and attends over the stored ones. The cache costs `2 × L × n_kv_heads × hd × 4 bytes` per token in fp32 (P1.2 does this arithmetic for real models). **GQA** means `n_kv_heads < n_heads`: query head `h` reads KV head `h / (n_heads / n_kv_heads)`. That shrinks the cache by the same factor.

### Sampling

- **Greedy:** argmax.
- **Temperature T:** divide the logits by T before the softmax. T < 1 sharpens, T > 1 flattens.
- **Top-p (nucleus):** sort the probabilities, keep the smallest prefix whose mass is ≥ p, renormalize, and sample. Pass the uniform random number `u` in explicitly so that tests can be deterministic.

The handbook's TopPvsTopK interactive (`docs/model-interaction/inference-parameters.md` in `modular/llm-inference-handbook`) shows the effect of each knob.

## 2. Where the time goes: predict first

For one decode step, every weight is read **once** from memory and used in **one** multiply-add. Arithmetic intensity is about 2 FLOP / 4 bytes = 0.5 (fp32), far left of your P0.4 ridge point. So:

```
decode tokens/s  ≲  memory bandwidth  ÷  bytes of weights read per token
```

> **Predict first (the worked prediction).** Take your measured all-threads triad bandwidth **B** from P0.4 `03_stream` (results/stream.json), and the model's weight bytes **W** (the engine prints them at start-up, `weights … MiB`). Predicted ceiling = B / W tokens/s.
>
> *Worked example with made-up round numbers. Replace them with yours:* if B = 40 GB/s and the model is a 135M-parameter Llama in fp32 (135e6 × 4 B = 0.54 GB), the ceiling is 40 / 0.54 ≈ **74 tokens/s**. Run `bench/engine_bench.py` and compare. Expect to land below the ceiling, because of:
> - thread fork/join per matvec (tiny matrices make this significant)
> - the KV-cache reads, which grow with position
> - the LM head, which at small d is a large share of the weights
> - the non-matvec ops

For the tiny random model used in tests (a few hundred KB), the weights fit in L2/L3. Then the bound is cache bandwidth and threading overhead, not DRAM. That is a nice illustration that the roofline's "B" depends on where the data lives.

## 3. Profile it

```bash
perf record -g ./build/engine-v0/s2s-engine --model build/my-model --steps 64 && perf report
```

Expect `ops::matvec` (inside the thread pool workers) to dominate. If `std::function` / pool overhead shows up high for a tiny model, that is the fork/join cost per matvec. Exercise 5 and P6 fix it differently: by batching, and by fusing work into fewer, larger tasks.

---

## Walkthrough

```bash
cmake -S platform/engine/v0 -B build/engine-v0 -DCMAKE_BUILD_TYPE=Release && cmake --build build/engine-v0 -j
uv run python platform/engine/v0/tools/make_tiny_llama.py build/tiny-llama
./build/engine-v0/s2s-engine --model build/tiny-llama --prompt-ids "1 2 3 4" --steps 32
uv run python platform/engine/v0/reference/llama_numpy.py --model build/tiny-llama --prompt "1 2 3 4" --steps 32
uv run pytest platform/engine/v0/tests     # teacher-forced logits + greedy tokens, engine vs NumPy
```

Optional, real text (needs network + the `torch` extra; check the model's license):

```bash
uv run --extra torch python platform/engine/v0/tools/convert_hf_model.py <a-small-LlamaForCausalLM> build/my-model
uv run --extra torch python platform/engine/v0/tools/chat.py --model build/my-model "Once upon a time"
```

## What you should see

- The two greedy runs on `build/tiny-llama` print **the same token ids**. The test suite also allows a divergence at an exact near-tie (≤ 1e-4 logit gap), which is legitimate fp32 noise, and reports it.
- `pytest platform/engine/v0/tests` passes (2 tests).
- The engine's stderr line prints weights MiB, KV cache MiB, and prefill and decode tok/s. **TODO(run): record yours** for the tiny model and (optionally) a converted real model, next to your B / W prediction.

## Exercises

You write the ops; the engine around them is given. The exercises build `platform/engine/v0` against **your** `exercises/ops/ops.hpp`.

```bash
uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/exercises/make_fixtures.py build/p05-fixtures
cmake -S course/P0-systems-primer/P0.5-project-tiny-engine-cpu/exercises -B build/p05-ex && cmake --build build/p05-ex -j
ctest --test-dir build/p05-ex --output-on-failure
# reference: add -DS2S_USE_SOLUTIONS=ON (uses platform/engine/v0/src/ops.hpp)
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [RMSNorm and RoPE](exercises/01-rmsnorm-rope.md) | easy | vs NumPy fixtures, `atol=1e-5` |
| 2 | [Attention with a KV cache (GQA)](exercises/02-attention.md) | medium | vs NumPy fixtures at every position, `atol=1e-5` |
| 3 | [Full forward pass](exercises/03-full-forward.md) | medium | your engine's teacher-forced logits vs NumPy (`rtol=atol=1e-4`) + greedy tokens |
| 4 | [Top-p sampling](exercises/04-top-p.md) | medium | χ² goodness of fit over 10⁵ seeded draws, plus edge cases |
| 5 | [Int8 weights (`runq`-style)](exercises/05-int8-weights.md) | hard | q8 matvec within its derived error bound; perplexity delta report |

## Benchmark

```bash
uv run python course/P0-systems-primer/P0.5-project-tiny-engine-cpu/bench/engine_bench.py --model build/tiny-llama
```

It prints `model | threads | prompt toks | prefill tok/s | decode tok/s | predicted decode ceiling (B/W) | % of ceiling` and writes `results/engine.json`. B comes from P0.4's `results/stream.json` if present; otherwise the script tells you to run `03_stream` first.

## Common mistakes

- **Wrong RoPE convention** (interleaved vs rotate_half). Layer-0 outputs already differ from the reference.
- **Off-by-one in causal attention.** Attend to positions `0..pos` *inclusive*: the current token sees itself.
- **GQA head mapping.** It is `h / group`, not `h % n_kv_heads`.
- **Softmax without max-subtraction.** `exp(90)` overflows float. Subtract the max.
- **Applying RoPE to V.** Only q and k are rotated.
- **Re-initializing the KV cache per step.** Then the "cache" is empty and the model sees one token at a time.

## Go deeper

- llama2.c `run.c` (`karpathy/llama2.c@350e04fe`): `forward()`, `sample_topp()`. Compare its interleaved RoPE with ours.
- HF `src/transformers/models/llama/modeling_llama.py` (`huggingface/transformers@14e738b5`): `LlamaRMSNorm`, `rotate_half` / `apply_rotary_pos_emb`, `repeat_kv`.
- Modular handbook, *How does LLM inference work* (with the AutoregressiveDecodeStepper interactive).
- Silicon to Scale ch. 2 (transformer deep dive) and ch. 6 (inference systems).

## Go down when…

- A matvec is slow: [P0.4](../P0.4-simd-and-roofline/README.md).
- The thread pool shows up in `perf`: [P0.3](../P0.3-threads-atomics-caches/README.md).
- Loader errors: [P0.2](../P0.2-virtual-memory-and-mmap/README.md).

**Next:** [P1.1 Transformer forward pass, by the numbers](../../P1-inference-fundamentals/P1.1-transformer-forward-pass/README.md).

## Animations

Reuses [`p0-vm-page-walk.html`](../../../animations/p0-vm-page-walk.html) (first touch of the weights) and [`roofline.html`](../../../animations/roofline.html) (preset "LLM decode, batch 1").
