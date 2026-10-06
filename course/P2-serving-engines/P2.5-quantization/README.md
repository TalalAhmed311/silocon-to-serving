# P2.5 — Quantization (+ #7 bakeoff)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the number formats, the quantization math and the table building; ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for calibration, serving and evals. **Use an L4 (`g6.xlarge`) or newer for FP8.** T4 has no FP8 or bf16 ([aws.md](aws.md)) |
| **Time** | ≈40 min reading + ≈12 h hands-on (≈6 GPU-hours) |
| **Prerequisites** | P0.4 (int8 dot products), P1.4 (FP8 throughput), P2.1, P2.3 (#4) |
| **Pinned** | LLM Compressor `0.14.0`, GPTQModel `v7.5.0`, lm-evaluation-harness `v0.4.13`, vLLM `v0.31.0` |

## Learning objectives

1. Know the formats (FP16, BF16, FP8 E4M3/E5M2, INT8, INT4), their ranges and precisions, and which GPUs accelerate which.
2. Distinguish **weight-only** quantization (W4A16, W8A16) from **weight + activation** (W8A8, FP8) and from **KV-cache** quantization, and what each speeds up.
3. Explain how **AWQ** and **GPTQ** choose their scales, at the level of their papers.
4. Produce FP8, AWQ and GPTQ checkpoints with LLM Compressor, serve them with vLLM, and run **#7**: quality vs latency vs VRAM in one table.

## Why this matters

Decode is memory-bound (P1.2), so **bytes per weight is speed**. W4A16 reads ~4× fewer weight bytes than BF16 and FP8 reads 2× fewer, and the KV cache works the same way. Quantization is also how a model that doesn't fit, fits. But every format costs some quality, and the cost depends on the model, the task and the method. You only find out by measuring, which is what #7 does.

---

## 1. The formats

| Format | Bits (sign/exp/mantissa) | Max | Precision (relative step near 1) | Accelerated on |
|---|---|---|---|---|
| FP16 | 1/5/10 | 65,504 | 2⁻¹⁰ ≈ 1e-3 | tensor cores since Volta |
| BF16 | 1/8/7 | ≈3.4e38 (FP32 range) | 2⁻⁷ ≈ 8e-3 | Ampere+ (not T4) |
| FP8 E4M3 ("fn": no inf) | 1/4/3 | 448 | 2⁻³ = 0.125 | Ada (L4, L40S), Hopper, Blackwell |
| FP8 E5M2 | 1/5/2 | 57,344 | 2⁻² | same; used for gradients, rarely for inference |
| INT8 | integer + scale | 127 × scale | uniform: scale | IMMA since Turing |
| INT4 | integer + scale | 7 × scale | uniform: scale | weight-only on most serving kernels (dequantized to fp16 in the GEMM) |

`examples/01_formats.py` quantizes the same tensor to each format and prints the round-trip error. The FP8 rows run in emulation on a laptop.

## 2. What gets quantized, and what that buys

| Scheme | Weights | Activations | Speeds up | Typical tools |
|---|---|---|---|---|
| **W4A16** (AWQ, GPTQ) | int4 + group scales | fp16/bf16 | memory-bound decode (~4× fewer weight bytes); not prefill | LLM Compressor, GPTQModel |
| **W8A8-INT8** | int8 | int8 (per-token dynamic) | decode, and prefill via int8 tensor cores | LLM Compressor (SmoothQuant + GPTQ) |
| **FP8** (W8A8-FP8) | fp8 (per-channel) | fp8 (per-token dynamic) | both: 2× fewer bytes *and* 2× tensor FLOPs on Ada/Hopper | LLM Compressor `FP8_DYNAMIC` (no calibration data needed) |
| **FP8 KV cache** | — | KV in fp8 | ~2× KV capacity, fewer attention bytes | vLLM `--kv-cache-dtype fp8` |

**Group-wise scales.** One scale per 128 weights (`group_size=128`) instead of per row lets each group follow its own range. Outliers then hurt less. Each scale costs 16 bits per 128 weights, about 0.125 extra bits per weight.

## 3. AWQ and GPTQ in one paragraph each

- **GPTQ** (Frantar et al., 2210.17323) quantizes the weights of a layer one column at a time. After each column is rounded, it **updates the remaining columns** to compensate for the error, using second-order (Hessian) information from calibration activations. The rounding errors partly cancel instead of accumulating.
- **AWQ** (Lin et al., 2306.00978) observes that a small fraction of weight channels matter much more, namely those multiplied by large activations. Rather than keeping them in high precision, it **scales those channels up** before quantization and folds the inverse scale into the previous op, so they lose less relative precision. The scales are searched on calibration data.

Both need a few hundred calibration samples. FP8 dynamic needs none: FP8 has enough range that per-channel scales computed from the weights alone suffice.

> **Predict first.** On an L4 (≈300 GB/s, UNVERIFIED), with an 8B model and batch-1 decode:
> - bf16 ceiling ≈ 18.7 tok/s (P1.4)
> - FP8 ≈ 37 tok/s (half the bytes)
> - W4A16 ≈ 70 tok/s on paper (≈ a quarter of the bytes plus scales)
>
> In practice W4A16 lands well below its paper ceiling, because dequantization costs compute inside the GEMM. Predict where each variant will sit as a % of its own ceiling, then measure with #7.

## 4. #7: the bakeoff

`platform/bakeoff` runs, for each variant in `variants.yaml`:

1. start vLLM
2. record VRAM and KV-cache tokens
3. run #4 at fixed rates
4. run lm-eval tasks against the live server
5. stop

`table.py` assembles the comparison table. Quality numbers come with a sample count (`limit`), because 250 samples per task detects large regressions, not 0.5-point ones. Say so in the report.

---

## Walkthrough

```bash
D=course/P2-serving-engines/P2.5-quantization
uv run python $D/examples/01_formats.py                          # T0
# T2 (aws.md): quantize, serve, bakeoff
python platform/bakeoff/quantize.py --base /opt/models/base --scheme fp8-dynamic --out /opt/models/base-fp8
python platform/bakeoff/run.py platform/bakeoff/variants.yaml
python platform/bakeoff/table.py results/bakeoff                  # T0 once results are copied back
```

## What you should see

- `01_formats.py` prints round-trip error per format on Gaussian data with a few outliers. Computed, so deterministic (seeded). Expected shape: FP16 is the most accurate and INT4 per-tensor the least. Group-wise INT4 is far better than per-tensor INT4, and E4M3 sits between INT8 and INT4.
- #7 (`TODO(run-on: g6.xlarge)`): FP8 has VRAM ≈ half of bf16's weights, decode tok/s ≈ 1.6–1.9× and a small quality delta. W4A16 has the smallest VRAM and a decode speedup below 4×. The FP8 KV cache gives ≈ 2× KV tokens.

## Exercises

```bash
uv run pytest course/P2-serving-engines/P2.5-quantization/exercises
```

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [INT4 group quantization in NumPy](exercises/01-int4-group.md) | T0 | error bounds; symmetric vs asymmetric; group size effect |
| 2 | [FP8 E4M3 emulation](exercises/02-fp8-emulation.md) | T0 | bit-exact vs `torch.float8_e4m3fn` (CPU) over all in-range values |
| 3 | [The #7 table from results](exercises/03-bakeoff-table.md) | T0 (+ T2 run) | `table.py` on fixture results; then your real run |
| 4 | [FP8 KV cache: capacity vs quality](exercises/04-fp8-kv.md) | T2 | `check_fp8kv.py` on your recorded JSON |

## Common mistakes

- **Comparing quality on 50 samples** and declaring a 1-point drop. That is noise.
- **FP8 on a T4.** It silently falls back or fails.
- **Quantizing `lm_head` and the embeddings** by default. They are sensitive and cheap to keep in bf16 (`ignore=["lm_head"]`).
- **Forgetting that W4A16 doesn't speed up prefill.** Prefill is compute-bound, and the activations are still 16-bit.
- **Calibrating on data unlike your traffic.** For example, code-heavy traffic and a chat calibration set.

## Go deeper

- AWQ (2306.00978) · GPTQ (2210.17323) · FP8 Formats for Deep Learning (2209.05433).
- LLM Compressor `examples/` at tag 0.14.0 (FP8 dynamic, W4A16 AWQ/GPTQ, W8A8).
- vLLM `vllm/model_executor/layers/quantization/`: how each format is dispatched to a kernel.
- Modular handbook: *LLM quantization*.
- Silicon to Scale ch. 8 (quantization).
- llama.cpp block formats (`ggml/src/ggml-common.h`), the CPU-side cousins of what you built in P0.4/P0.5.

**Next:** [P2.6 Speculative decoding and CUDA graphs](../P2.6-speculative-decoding-and-cuda-graphs/README.md).
