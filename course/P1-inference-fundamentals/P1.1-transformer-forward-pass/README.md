# P1.1 — Transformer forward pass, by the numbers

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). Example 02 needs the `torch` extra (CPU only) |
| **Time** | ≈35 min reading + ≈6 h hands-on |
| **Prerequisites** | P0.5 (you've implemented every op once already) |
| **You will build** | an instrumented NumPy Llama that counts its own FLOPs and bytes; parity with Hugging Face `transformers` |

## Learning objectives

1. Write a Llama-style decoder forward pass in NumPy (RMSNorm, RoPE, GQA, SwiGLU). You did this in P0.5; now you *account* for it.
2. Count parameters, FLOPs and bytes moved per op as functions of `(d, L, H, H_kv, d_ff, V)`.
3. Prove your NumPy model is numerically identical to HF `LlamaForCausalLM` on a random-init config.
4. Explain why **≈ 2·P FLOPs per token** is a good estimate, and when attention makes it wrong.

## Why this matters

Every capacity plan (P1.5), roofline placement (P1.4), quantization trade-off (P2.5) and TP choice (P4.2) is arithmetic on these formulas. Being able to derive "an 8B model at batch 1 reads 16 GB per token in fp16" in your head is the core skill of an inference engineer.

---

## 1. Notation

| Symbol | Meaning | Llama-3-8B (from its `config.json`) |
|---|---|---|
| `d` | hidden size (`hidden_size`) | 4096 |
| `L` | layers (`num_hidden_layers`) | 32 |
| `H` | query heads (`num_attention_heads`) | 32 |
| `H_kv` | key/value heads (`num_key_value_heads`) | 8 |
| `h = d / H` | head dim | 128 |
| `d_ff` | MLP width (`intermediate_size`) | 14336 |
| `V` | vocabulary (`vocab_size`) | 128256 |

> **Verify before you cite.** The right column is the published `config.json` of `meta-llama/Meta-Llama-3-8B` as the author remembers it. Exercise 1 makes you load the real file (through `transformers.AutoConfig`, or by reading `config.json` from a local snapshot) and assert each value. Until you have, treat the column as UNVERIFIED.

## 2. Parameters

Per layer:

| Tensor | Shape | Params |
|---|---|---|
| `q_proj` | `[H·h, d]` | `d²` |
| `k_proj`, `v_proj` | `[H_kv·h, d]` each | `2·d·H_kv·h` |
| `o_proj` | `[d, H·h]` | `d²` |
| `gate`, `up` | `[d_ff, d]` each | `2·d·d_ff` |
| `down` | `[d, d_ff]` | `d·d_ff` |
| 2 × RMSNorm | `[d]` | `2d` |

Per layer: `P_layer = 2d² + 2·d·H_kv·h + 3·d·d_ff + 2d`
Total: `P = L·P_layer + V·d (embed) + V·d (head, unless tied) + d (final norm)`

Plugging in the 8B column: per layer `2·4096² + 2·4096·1024 + 3·4096·14336 + 8192 ≈ 218M`, × 32 = 6.98B, plus 2 × 128256 × 4096 = 1.05B of embeddings and head, ≈ **8.03B**. It matches the name, which is a good sign the formula is right. `examples/03_flop_counter.py` computes it from any config.

## 3. FLOPs per token (decode)

A matvec with an `[m, n]` matrix costs `2mn` FLOPs (a multiply and an add per weight). Every weight matrix is used once per token, so the **weight FLOPs per token ≈ 2 × (params in matrices)**. That is where "2P" comes from. The parts that are *not* weights:

- **Attention scores and values.** At context length `t`, each layer computes `q·Kᵀ` (`2·H·h·t`) and `softmax·V` (`2·H·h·t`), which is `4·d·t` per layer. That is negligible at t = 100 and dominant at t = 100k: for the 8B model, attention FLOPs (`4·d·L·t`) equal the weight FLOPs (`2P`) at `t = 2P / (4·d·L) = P / (2·d·L) ≈ 8.03e9 / (2·4096·32) ≈ 31k` tokens. (GQA doesn't help here: it shrinks K/V *storage*, but every query head still does its own dot products.)
- Norms, RoPE, softmax and SiLU: O(d) per layer, negligible.

`FLOPs/token ≈ 2·P_matrices + 4·d·L·t`

## 4. Bytes per token (decode, batch 1)

Every weight byte is read once: `bytes ≈ P × bytes_per_param` (2 for fp16/bf16, 1 for fp8/int8, ~0.5 for int4). On top of that come the KV-cache reads, `2 · L · H_kv · h · t · bytes_kv`, which grow with context. Activations are tiny.

Arithmetic intensity at batch 1 ≈ `2P / (2P bytes)` = **1 FLOP/byte in fp16**. The 2 in the numerator is FLOPs per weight, the 2 in the denominator is bytes per weight. That is why decode is memory-bound on every accelerator (P1.4).

## 5. Prefill is different

With `T` prompt tokens processed together, each weight is read **once** and used `T` times, so intensity ≈ `T` FLOP/byte. Prefill is a GEMM. For T in the hundreds it is compute-bound. Same weights, same FLOPs per token, opposite bottleneck. P1.2 builds on exactly this.

## 6. HF parity: why bother?

Your NumPy model is the reference for *everything*: the C++ engine (P0.5), the CUDA engine (P6.7), and the custom kernels (P5). If it is wrong, every test downstream is wrong in the same way and still passes. `examples/02_hf_parity.py` builds a tiny random-init `LlamaConfig`, saves it, loads the identical weights into `LlamaNumpy`, and compares logits at every position. The test checks that the max |Δ| is below 1e-4.

---

## Walkthrough

```bash
uv run python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/01_numpy_llama_counted.py
uv run --extra torch python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/02_hf_parity.py
uv run python course/P1-inference-fundamentals/P1.1-transformer-forward-pass/examples/03_flop_counter.py --preset llama3-8b
```

## What you should see

- `01`: per-op counters for one decode step of the tiny model. The matmul rows hold > 99% of the FLOPs. These are counts, so they are exact and identical on every machine.
- `02`: `max |logits_numpy − logits_hf| < 1e-4` at all positions, then `PARITY OK`.
- `03`: a parameter breakdown table and FLOPs/token at t = 1k, 8k, 32k, 128k, with the attention share growing with t. Computed from the config's numbers, not measured.

## Exercises

```bash
uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises
S2S_SOLUTIONS=1 uv run pytest course/P1-inference-fundamentals/P1.1-transformer-forward-pass/exercises
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Parameter count formula](exercises/01-param-count/README.md) | easy | vs `sum(p.size)` on 4 random configs (and vs HF if the `torch` extra is installed) |
| 2 | [FLOPs per token](exercises/02-flops/README.md) | easy | vs the instrumented counter in example 01 |
| 3 | [GQA KV saving](exercises/03-gqa/README.md) | easy | KV bytes per token for MHA, GQA, MQA |
| 4 | [RoPE's relative-position property](exercises/04-rope-relative/README.md) | hard | ⟨R(m)q, R(n)k⟩ depends only on m − n, numerically |

## Common mistakes

- **Forgetting the LM head.** `V·d` is 0.5B params for Llama-3-8B, and it is untied there, so it is read every token.
- **Counting the embedding as FLOPs.** It is a lookup (one row), not a matvec.
- **Using `num_attention_heads` for K/V.** With GQA, K/V use `num_key_value_heads`.
- **"FLOPs = 2P" at 100k context.** At long context, attention dominates (§3).
- **Mixing up bytes/param** for the stored dtype vs the compute dtype. An int4 model may compute in fp16, but it *reads* 0.5 B/param.

## Go deeper

- kipply, *Transformer Inference Arithmetic*: the canonical derivation of these formulas, plus KV-cache and latency math.
- nanoGPT `model.py` (`karpathy/nanoGPT@3adf61e1`): a compact GPT forward pass; compare `CausalSelfAttention` with our GQA.
- llm.c `train_gpt2.c` (`karpathy/llm.c@f1e2ace6`): the same forward pass in C, with explicit buffers.
- HF `modeling_llama.py`: `LlamaAttention.forward`, `apply_rotary_pos_emb`, `repeat_kv`.
- Silicon to Scale ch. 1 (performance mindset) and ch. 2 (transformer deep dive).
- Modular handbook: *What is LLM inference*, with the RequestLifecycle interactive.

## Go down when…

An op is unclear → [P0.5](../../P0-systems-primer/P0.5-project-tiny-engine-cpu/README.md). **Next:** [P1.2 Prefill vs decode, and the KV cache](../P1.2-prefill-decode-kv-cache/README.md).
