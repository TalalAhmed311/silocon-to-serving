# P4.1: NCCL collectives and bandwidth math

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for the ring simulator, gloo collectives, cost model and parser. ![T3](https://img.shields.io/badge/tier-T3%20multi--GPU-red) for nccl-tests on a 4-GPU PCIe box and (optionally) an 8-GPU NVLink box |
| **Time** | ≈30 min reading + ≈10 h hands-on |
| **Prerequisites** | P1.4 (rooflines: communication is "just another roofline"), P0.3 (threads, barriers) |
| **You will build** | a tested α–β communication model (`commmodel.py`) that P4.2 uses to predict tensor-parallel cost |

## Learning objectives

1. State what broadcast, reduce, all-reduce, reduce-scatter, all-gather and all-to-all do, and who ends up with what.
2. Derive the ring all-reduce cost `2(n−1)·α + 2(n−1)/n · S/B`, and say when a tree beats it.
3. Convert nccl-tests' **algbw** to **busbw** and compare it with a cited link peak.
4. Predict the time of a tensor-parallel layer's all-reduce from its activation size and the link bandwidth.
5. Read `NCCL_DEBUG=INFO` and `nvidia-smi topo -m` to see which links NCCL actually uses.

---

## 1. The collectives

| Collective | Before (rank r) | After (every rank, unless noted) | Used by |
|---|---|---|---|
| broadcast | root has x | x | weight distribution |
| reduce | x_r | root has Σ x_r | metrics |
| **all-reduce** | x_r | Σ x_r | DP gradients, **TP activations** |
| **reduce-scatter** | x_r | rank r has chunk r of Σ x_r | FSDP gradients |
| **all-gather** | chunk c_r | concat(c_0 … c_{n−1}) | FSDP params, sequence parallel |
| **all-to-all** | n blocks | rank r gets block r from everyone | MoE expert parallelism |

## 2. Ring all-reduce and its cost

Split each rank's buffer into n chunks. Arrange the ranks in a ring.

1. **Reduce-scatter**, n−1 steps: each rank sends one chunk right and adds the chunk it receives. Afterwards, each rank owns one fully summed chunk.
2. **All-gather**, n−1 steps: circulate the summed chunks until everyone has all of them.

Each step moves S/n bytes per rank, and all links are busy at once. Total time:

```
T_ring = 2(n−1)·α  +  2(n−1)/n · S/B            α = per-step latency, B = per-link bandwidth
```

The bandwidth term **barely grows with n**, so ring is bandwidth-optimal. The latency term grows linearly, so for small messages on many ranks a **tree** (`≈ 2 log₂n · α + 2S/B`) wins. NCCL chooses between them per call. See `commmodel.crossover_size`.

The animation [`p4-ring-allreduce.html`](../../../animations/p4-ring-allreduce.html) steps through both phases chunk by chunk.

## 3. algbw vs busbw

nccl-tests prints `algbw = S/t`. For an all-reduce, each rank moves `2(n−1)/n · S` bytes, so the link actually carried `busbw = algbw × 2(n−1)/n`. **busbw is the number you compare with the link's peak**, whatever n and whatever the collective. The factors are in `commmodel.BUSBW_FACTOR` (from nccl-tests `doc/PERFORMANCE.md`).

## 4. Links

| Link | Where | What to cite |
|---|---|---|
| PCIe Gen4/Gen5 x16 | `g6.12xlarge`-class (4× L4, no NVLink) | PCIe spec bandwidth per direction (UNVERIFIED until cited). GPU↔GPU traffic may cross the CPU root complex |
| NVLink + NVSwitch | `p4d.24xlarge` (8× A100), `p5` (8× H100) | the A100/H100 whitepaper NVLink bandwidth per GPU (UNVERIFIED) |
| EFA (inter-node) | p4d/p5 | the EC2 instance network bandwidth (UNVERIFIED) |

`nvidia-smi topo -m` shows the path between every GPU pair (`PIX`, `PHB`, `SYS`, `NV#`). `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` prints the rings and trees NCCL built, and whether it used `P2P`, `SHM` or `NET`.

> **Predict first.** Tensor parallelism with TP = 4 on Llama-3-8B-shaped layers: d_model = 4096, fp16, batch × tokens = 1 during decode. Each transformer block does **2 all-reduces** of `1 × 4096 × 2 B = 8 KB`. At 8 KB, is the cost α-dominated or B-dominated? With your fitted α from exercise 2, how many microseconds per token does TP communication add across 32 layers? Compare with the decode step time from P2. P4.2 measures it.

---

## Walkthrough

```bash
uv run python course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/examples/01_collectives_numpy.py --ranks 4
uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/examples/02_torch_dist_gloo.py --world 4
uv run pytest course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/exercises
```

The T3 path is in [aws.md](aws.md).

## What you should see

- `01`: `ring all-reduce OK: 4 ranks, 6 steps (2(N-1)), each rank sent 1.50× its buffer`.
- `02` (CPU loopback): checks pass, and the time per all-reduce is roughly flat below ~64 KB and then grows linearly. That's α and β, visible even on one machine.
- nccl-tests (`TODO(run-on: g6.12xlarge, p4d.24xlarge)`): busbw rises with size and plateaus. The plateau as a fraction of the link peak is your headline number. Expect the PCIe box to plateau far below the NVLink box.

## Bench

| instance | collective | plateau busbw (GB/s) | link peak (cited) | % | fitted α (µs) |
|---|---|---|---|---|---|
| g6.12xlarge (4× L4, PCIe) | all_reduce | `TODO(run-on: g6.12xlarge)` | | | |
| p4d.24xlarge (8× A100, NVLink) | all_reduce | `TODO(run-on: p4d.24xlarge)` | | | |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Ring all-reduce simulator: correctness, 2(n−1) steps](exercises/01-ring.md) | T0 | `test_ring.py` |
| 2 | [Fit the α–β model to nccl-tests output](exercises/02-alpha-beta.md) | T0 → T3 | `test_comm_model.py` |
| 3 | [busbw from algbw for each collective](exercises/03-busbw.md) | T0 | `test_comm_model.py` |
| 4 | *(hard)* [Reduce-scatter + all-gather = all-reduce, in the simulator and in the timing](exercises/04-rs-ag.md) | T0 + T3 | `test_ring.py::test_rs_then_ag_equals_all_reduce` |

## Common mistakes

- Comparing **algbw** with the link peak. That undercounts all-reduce traffic by ~2×.
- Fitting one α–β line to all sizes. Large messages dominate the least-squares fit and hide α.
- Measuring the first iteration: NCCL builds communicators and rings lazily. Use warm-up (`-w`).
- Assuming GPUs on a PCIe box talk directly. Check `topo -m`: a `SYS` path crosses CPU sockets.

## Go deeper

- NCCL (`v2.32.3-1`) and nccl-tests (`v2.21.1`) docs, especially `doc/PERFORMANCE.md`.
- *The Ultra-Scale Playbook* (Hugging Face), the communication chapters. *How to Scale Your Model* (JAX team), "All about rooflines" and the collectives section.
- Silicon to Scale ch. 14 (networking). AWS EFA docs.

**Next:** [P4.2 Parallelism strategies](../P4.2-parallelism-strategies/README.md).
