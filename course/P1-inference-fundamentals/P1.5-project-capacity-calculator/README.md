# P1.5 — Project: capacity calculator (D3)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f). Exercise 3's comparison with vLLM's log is T2 |
| **Time** | ≈20 min reading + ≈9 h building |
| **Prerequisites** | P1.1–P1.4 |
| **You will build** | `platform/capacity`: a tested CLI that predicts VRAM, KV capacity, max batch, decode ceilings and prefill time for any model × GPU |

## Learning objectives

1. Combine P1.1 (parameters), P1.2 (KV), P1.3 (metrics) and P1.4 (rooflines) into one tool with explicit assumptions.
2. Predict, *before* renting a GPU, whether a deployment fits and roughly how it will perform.
3. Know which inputs dominate the answer: weight dtype, KV dtype, context and TP. Know which are guesses: utilization factors.

## What the calculator computes

| Output | Formula | Lesson |
|---|---|---|
| weights per GPU | `total_params × bytes(weight_dtype) / TP` | P1.1 |
| KV budget per GPU | `memory × mem_util − weights − activation reserve` | P1.2 |
| KV bytes per token | `2 · L · H_kv · h · bytes(kv_dtype)`, sharded over TP | P1.2 |
| max sequences at context c | `⌊KV budget / (c × KV bytes per token)⌋` | P1.2 |
| decode ceiling at batch B | `bw_util × bandwidth × TP ÷ (active weight bytes + B × c/2 × KV bytes per token)` | P1.2, P1.4 |
| prefill time for T tokens | `(2·active_params·T + 2·L·H·h·T²) ÷ (mfu × peak × TP)` | P1.1, P1.4 |

Three **named assumptions** sit outside the physics. They are printed with every result and are yours to calibrate in P2:

- `mem_util` (default 0.90): vLLM's `--gpu-memory-utilization` default.
- `bw_util` (default 0.80): achievable fraction of peak bandwidth. P1.4 exercise 4 measures it.
- `mfu` (default 0.50): model-FLOPs utilization for prefill GEMMs.

**MoE models** read only `experts_per_token` experts' MLP weights per token, but must *store* all of them. Mixtral-style configs report much higher total than active params. The calculator uses total params for VRAM and active params for decode bytes.

> **Predict first, with the tool.** Before P2.1, run:
> ```bash
> PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4
> PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4 --weights fp8 --kv fp8
> ```
> Write down the max sequences at 4k context and the batch-1 decode ceiling. In P2.1 you start vLLM on a `g6.xlarge` and compare against the KV-cache size it logs at startup (exercise 3). Any gap is your calibration of `mem_util` and the activation reserve.

## Walkthrough

```bash
PYTHONPATH=platform uv run python -m capacity.cli --model llama3-8b --gpu L4
PYTHONPATH=platform uv run python -m capacity.cli --model llama3-70b --gpu H100-SXM --tp 4 --weights fp8 --kv fp8 --context 8192
PYTHONPATH=platform uv run python -m capacity.cli --model mixtral-8x7b --gpu L40S --tp 2
uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises
```

## What you should see

Tables computed from the presets and `gpu_specs.yaml`. **Both are UNVERIFIED**, and the CLI prints that next to the GPU name. Expected shapes:

- an 8B bf16 model fits on a 24 GB GPU with a few GiB of KV left; fp8 roughly doubles the sequences that fit
- 70B bf16 does **not** fit on one 80 GB GPU, but fits on 4 with TP
- the decode ceiling falls as the batch grows *per sequence*, while aggregate tokens/s rises
- prefill time grows faster than linearly at 8k tokens, from the T² attention term

## Exercises: the tests are the spec

Build your own `calculator.py` (in `exercises/`), then compare it with the reference in `platform/capacity/core.py`.

```bash
uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises            # your calculator.py
S2S_SOLUTIONS=1 uv run pytest course/P1-inference-fundamentals/P1.5-project-capacity-calculator/exercises  # reference
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | Weights bytes = safetensors size | easy | exact vs the sum of tensor sizes in tiny generated checkpoints (fp32 and tied); real-model check is `TODO(run)` |
| 2 | KV bytes per token, max sequences | easy | hand-computed |
| 3 | Predicted vs vLLM-reported KV capacity | medium (T2) | parser unit test; the comparison is `TODO(run-on: g6.xlarge)` |
| 4 | Cross-check against the Modular handbook's GPU memory calculator | medium | manual: fill in `exercises/crosscheck.md` for 3 configs and explain the differences |
| 5 | MoE + TP | hard | Mixtral-shaped totals vs hand-derived numbers; TP halves weights and KV per GPU |

Exercise details are in [`exercises/README.md`](exercises/README.md).

## Common mistakes

- Using **total** params for MoE decode bandwidth: it overestimates the bytes by about 3.6× for Mixtral.
- Forgetting that the **LM head** is read every token even when tied.
- Ignoring **activation / CUDA-graph memory**. vLLM reserves memory for both before sizing the KV cache.
- Treating `bw_util` and `mfu` as facts. They are your calibration knobs.

## Go deeper

- Modular handbook: *Calculating GPU memory for LLMs* (its Calculator interactive is exercise 4's cross-check).
- Silicon to Scale ch. 11 (KV-cache engineering).
- vLLM docs: `--gpu-memory-utilization`, `--max-model-len` and `--kv-cache-dtype` in the engine arguments page (`vllm/docs/` at the pinned SHA).

**Next:** [P2.1 First serve with vLLM](../../P2-serving-engines/P2.1-first-serve-with-vllm/README.md), where every prediction meets a GPU.
