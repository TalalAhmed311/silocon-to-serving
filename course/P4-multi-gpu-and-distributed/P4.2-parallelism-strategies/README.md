# P4.2: Parallelism strategies (TP / PP / DP / EP, FSDP)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the TP MLP numerics, the pipeline simulator, ZeRO memory accounting and TP in the capacity calculator. ![T3](https://img.shields.io/badge/tier-T3%20multi--GPU-red) for vLLM TP = 1/2/4 and FSDP2 |
| **Time** | ≈35 min reading + ≈11 h hands-on |
| **Prerequisites** | P4.1 (collectives, α–β model), P1.5 (D3 capacity calculator), P1.1 (the MLP and attention shapes) |
| **You will build** | TP communication cost in D3, and a measured TP = 1/2/4 comparison you can explain number by number |

## Learning objectives

1. Split an MLP Megatron-style (column, then row) and show that it needs **one** all-reduce per block.
2. Compute the pipeline bubble `(p−1)/(m+p−1)`, and explain why 1F1B keeps the same bubble with far less activation memory.
3. Account for training memory under DDP, ZeRO-1/2/3 and FSDP, and name the collectives each one uses.
4. Know what expert parallelism costs: two all-to-alls per MoE layer.
5. For **serving**, choose between TP, PP and independent replicas using D3 plus P4.1's comm model, then check against vLLM.

---

## 1. Tensor parallelism

An MLP is `y = (silu(x W_g) ⊙ (x W_u)) W_d`. Split `W_g` and `W_u` by **columns**, so each rank gets f/tp of the intermediate units. Each rank can then apply `silu` and the product to its own slice with **no communication**: elementwise ops don't mix units. Split `W_d` by **rows** to match, and each rank produces a full-shape **partial** `y_r`. One all-reduce gives `y = Σ y_r`.

Attention splits the same way: heads are columns of W_q, W_k, W_v (each rank owns heads/tp heads and their KV cache), and W_o is row-split. **2 all-reduces per transformer layer** in total.

The other order (row-split first) would need a reduction *before* the nonlinearity. `parallel.tp_mlp_wrong_order` shows the error. The animation [`p4-tp-mlp.html`](../../../animations/p4-tp-mlp.html) walks through the split on 2 and 4 GPUs.

**For decode**, TP divides the weight bytes each GPU reads per token by tp: up to ~tp× faster per token, if communication is free. It isn't. Each token pays `2 · layers` all-reduces of `batch × d × 2 B`, which at batch 1 is ~8 KB. P4.1 showed those are **latency-bound**. `capacity.core.tp_comm_seconds_per_step` adds them to D3's decode ceiling.

## 2. Pipeline parallelism

Give each of p stages a contiguous slice of layers. A batch split into m microbatches flows through, and stage s sits idle while it waits for its first microbatch and again while it drains:

```
bubble = (p − 1) / (m + p − 1)          → small only when m ≫ p
```

**GPipe** runs all forwards, then all backwards, so stage 0 holds activations for all m microbatches. **1F1B** alternates one forward and one backward after a short warm-up: the same bubble, but at most p microbatches in flight. `parallel.simulate_pipeline` simulates both, and the tests check the formula against it.

For serving, PP adds a hop per stage to every token's latency, and it fills only with many concurrent requests. Use it to fit a model that's too big for one node's TP group, not to speed one up.

## 3. Data parallelism, ZeRO and FSDP

| Strategy | Bytes per param per GPU (mixed-precision Adam) | Collectives per step |
|---|---|---|
| DDP | 16 | all-reduce(grads) |
| ZeRO-1 | 4 + 12/n | all-reduce(grads), shard optimizer |
| ZeRO-2 | 2 + 14/n | reduce-scatter(grads) |
| ZeRO-3 / FSDP | 16/n | all-gather(params) before forward and backward + reduce-scatter(grads) |

FSDP moves about the same bytes as DDP (P4.1 exercise 4: RS + AG = AR), plus one extra parameter all-gather for the backward. It overlaps them with compute, layer by layer. `parallel.training_bytes_per_param` implements the table.

## 4. Expert parallelism

An MoE layer routes each token to k of E experts. With experts sharded across GPUs, each layer does an **all-to-all** to send tokens to their experts and another to bring results back. Its cost depends on the routing balance, which is why MoE serving cares about load-balancing losses and capacity factors (Silicon to Scale ch. 13).

## 5. Choosing for serving

| Situation | Choice | Why |
|---|---|---|
| model fits one GPU with KV headroom | **replicas** (TP = 1) | no comm. Throughput scales linearly with GPUs |
| model fits, but you need lower per-token latency | TP = 2–4 **within NVLink** | weights read per GPU drop by tp, and comm is cheap on NVLink |
| model doesn't fit one GPU | the smallest TP that fits with enough KV | each extra TP rank costs comm and wastes nothing else |
| model doesn't fit one node | TP within the node × PP across nodes | TP all-reduces need NVLink. PP's point-to-point sends tolerate EFA |

> **Predict first.** Llama-3-8B (bf16) on 4× L4 over PCIe. Use D3 with your P4.1 α and B to predict batch-1 decode tok/s at TP = 1, 2, 4, and aggregate tok/s for **4 × TP=1 replicas** versus **1 × TP=4**. Write the numbers down, then run `03_vllm_tp.sh`.

---

## Walkthrough

```bash
uv run python course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/01_tp_mlp_numpy.py
uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/02_tp_torch_gloo.py --tp 4
uv run pytest course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/exercises
PYTHONPATH=platform uv run python -c "from capacity import core; m=core.Model.from_file('platform/capacity/presets/llama3-8b.json'); \
  print({tp: core.plan(m,'L4',tp=tp,link_gbs=25,link_alpha_us=10).decode_ceiling[1] for tp in (1,2,4)})"   # 25 GB/s, 10 µs: placeholders → use your P4.1 fit
```

The T3 path is in [aws.md](aws.md).

## What you should see

- `01`: max abs error ~1e-6 at every tp, and a large error for the wrong order.
- pytest: TP numerics, the bubble formula equal to the simulation for both schedules, and 1F1B peak in-flight ≤ p.
- D3 with TP: batch-1 tok/s grows less than linearly with tp. With a large α it can even fall from TP = 2 to TP = 4.
- vLLM on 4× L4 (`TODO(run-on: g6.12xlarge)`): the same shape as D3, and per-GPU throughput that's best at TP = 1.

## Bench

| config | batch-1 ITL (ms) predicted | measured | knee out tok/s | tok/s per GPU |
|---|---|---|---|---|
| TP=1 (×4 replicas) | | `TODO(run-on: g6.12xlarge)` | | |
| TP=2 (×2 replicas) | | | | |
| TP=4 | | | | |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [TP MLP numerics match unsplit (fp32, atol 1e-5)](exercises/01-tp-mlp.md) | T0 | `test_tp_mlp.py` |
| 2 | [Pipeline bubble: formula vs simulation; 1F1B memory](exercises/02-pipeline.md) | T0 | `test_pipeline.py` |
| 3 | [Add TP communication to the D3 capacity calculator](exercises/03-tp-capacity.md) | T0 | `test_tp_capacity.py` |
| 4 | *(hard)* [Measured vLLM TP = 1/2/4 ITL vs the prediction, gaps explained](exercises/04-vllm-tp.md) | T3 | your bench table + write-up |

## Common mistakes

- Using TP across PCIe or nodes and expecting NVLink scaling.
- Measuring TP only at batch 1 (it helps latency) and concluding it helps throughput (it usually doesn't).
- Forgetting that TP also shards the KV cache (per-rank KV heads). With `kv_heads < tp`, vLLM replicates KV heads.
- Too few microbatches in PP: with m = p, the bubble is almost 50%.
- Counting FSDP as "extra communication": it's the same bytes as DDP, plus one parameter gather.

## Go deeper

- *The Ultra-Scale Playbook* (Hugging Face): TP, PP, ZeRO, EP chapters. *How to Scale Your Model*: the sharding and transformer-math chapters.
- Megatron-LM paper (Shoeybi et al.), ZeRO (Rajbhandari et al.), GPipe (Huang et al.), PipeDream 1F1B (Narayanan et al.). SOURCES.md §4 lists the IDs.
- torchtitan `v0.3.0` (FSDP2 + TP + PP in one codebase).
- Modular handbook: *Data, tensor, pipeline, expert, and hybrid parallelism*. Silicon to Scale ch. 10, 13.

**Next:** [P4.3 Distributed jobs, checkpointing, fault tolerance](../P4.3-distributed-jobs-and-checkpointing/README.md).
