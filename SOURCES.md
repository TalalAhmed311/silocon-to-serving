# SOURCES

Every external source the course uses: where it lives, the version it is pinned to, its license, whether it was verified, and which modules use it.

**Index date:** 2026-10-06. **Method:** `git ls-remote` for HEAD SHAs and tags, plus a blobless clone (`git clone --filter=blob:none`) to list file trees and read license files at the pinned SHA. Pinned commit = the default-branch HEAD on the index date. The full 40-character SHAs are in each link.

**Status key**

- **VERIFIED**: opened at the pinned SHA, paths confirmed.
- **VERIFIED (via repo)**: the website was blocked, but its source repo was read instead.
- **UNVERIFIED**: blocked by this build environment's egress policy or not checkable (books). Nothing is built on an UNVERIFIED source until it is checked. See [GAPS.md](GAPS.md).

> **License rule:** we link and summarize. No third-party code is vendored. Excerpts of 15 lines or fewer from permissively licensed code may appear, with source and license noted. **LeetGPU challenges are CC BY-NC-ND 4.0**, so we never reproduce problem statements, starters or tests. We only link to them, and our hint ladders and solutions are original.

## 1. Git repositories (all VERIFIED at the pinned SHA)

| Repo | Pinned commit | Latest release tag seen | License | Used by |
|---|---|---|---|---|
| [modular/llm-inference-handbook](https://github.com/modular/llm-inference-handbook/tree/5bddc17b973850cc9e3db47432f5bf9c66f644dd) | `5bddc17b9738` | — | Apache-2.0 | P1–P5 (anchor) |
| [karpathy/llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb) | `350e04fe3543` | — | MIT | P0.5 (#0 v0) |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/51ce9c11a6f2dfa895696c0048c4333e8953b728) | `51ce9c11a6f2` | b11453 (build tags; `v0.6.0` is a stale tag) | MIT | P0.4, P0.5, P2.5 |
| [huggingface/safetensors](https://github.com/huggingface/safetensors/tree/e246a2560645b7525f5775669ed816eb57c5bcc8) | `e246a2560645` | v0.8.0 | Apache-2.0 | P0.2, P0.5, P3.7 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291) | `3adf61e154c3` | — | MIT | P1.1 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c/tree/f1e2ace651495b74ae22d45d1723443fd00ecd3a) | `f1e2ace65149` | — | MIT | P1.1, P5.4–P5.5 |
| [huggingface/transformers](https://github.com/huggingface/transformers/tree/14e738b5d0cc69aa27a95dde272aea41fde44f2f) | `14e738b5d0cc` | v5.19.0 | Apache-2.0 | P1.1 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/d1f3d8b87083a95f914c674c477796950e2ba5fe) | `d1f3d8b87083` | v0.31.0 | Apache-2.0 | P2, P3.5, P5.10, P6 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/a34cba3c62ff9c0a3e89df9bb25943844804c6ba) | `a34cba3c62ff` | v0.5.21 | Apache-2.0 | P2.2, P6.4 |
| [vllm-project/guidellm](https://github.com/vllm-project/guidellm/tree/246f3b0f02053b4351f648093461d397ad534cc2) | `246f3b0f0205` | v0.8.0 | Apache-2.0 | P2.3, P2.4 (#4) |
| [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/tree/af206d991b518d454c4b056aa98bd90fa06a58fa) | `af206d991b51` | 0.14.0 | Apache-2.0 | P2.5 (#7) |
| [casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ/tree/88e4c76b20755db275574e6a03c83c84ba3bece5) | `88e4c76b2075` | v0.2.9 | MIT | P2.5 (history only — deprecated) |
| [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel/tree/d0e59f892b77228e6e9774fc4c43850410e362bb) | `d0e59f892b77` | v7.5.0 | Apache-2.0 | P2.5 (#7) |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520) | `d6de81643928` | v0.4.13 | MIT | P2.5 (#7) |
| [BerriAI/litellm](https://github.com/BerriAI/litellm/tree/5ed7ec851148cbb47d33c95e9264f19a0bf4d8e6) | `5ed7ec851148` | v1.104.0 | MIT | P2.7 (#6) |
| [envoyproxy/ai-gateway](https://github.com/envoyproxy/ai-gateway/tree/daa9f891a8afcb18576d4593ad18870d1d18abac) | `daa9f891a8af` | v1.1.0 | Apache-2.0 | P2.7 (#6) |
| [NVIDIA/gpu-operator](https://github.com/NVIDIA/gpu-operator/tree/cc6fd60084809b408fb6aad486c27f351b279aac) | `cc6fd6008480` | v26.7.1 | Apache-2.0 | P3.3, P4.4 |
| [NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter/tree/fafd151148052628061a80450b4ee037a5fa0c3c) | `fafd15114805` | 4.8.4 | Apache-2.0 | P3.5 (#2, #13) |
| [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin/tree/d5dce58aff4ddf572dc52666f949b37d6be2846c) | `d5dce58aff4d` | v0.20.1 | Apache-2.0 | P3.3, P4.4 |
| [kserve/kserve](https://github.com/kserve/kserve/tree/fddb8fefd53e0b9ec070f3cf86a2790f07424763) | `fddb8fefd53e` | v0.21.0 | Apache-2.0 | P3.4 |
| [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo/tree/b37890b6320e5775f18c4c1918cdd39cdb869ad8) | `b37890b6320e` | v1.5.0 | Apache-2.0 | P3.4 |
| [llm-d/llm-d](https://github.com/llm-d/llm-d/tree/23f5738838aa2f13aab5aad910f3e1a1692be9d6) | `23f5738838aa` | v0.10.0 | Apache-2.0 | P3.4 |
| [ray-project/ray](https://github.com/ray-project/ray/tree/1a9ba71a371d4dbcfc67116198d39ee9b20ab6e0) | `1a9ba71a371d` | ray-2.59.0 | Apache-2.0 | P3.4, P4.3 |
| [kedacore/keda](https://github.com/kedacore/keda/tree/8a0692a3e4548af86d4100f79dd71bd92584279e) | `8a0692a3e454` | v2.21.0 | Apache-2.0 | P3.6 (#3) |
| [terraform-aws-modules/terraform-aws-eks](https://github.com/terraform-aws-modules/terraform-aws-eks/tree/e07246207174bd1ad8ed250f3bb54a9494342e80) | `e07246207174` | v21.26.0 | Apache-2.0 | P3.2, infra/aws |
| [awslabs/data-on-eks](https://github.com/awslabs/data-on-eks/tree/cc5f8e8675531fb7f7aba6d585a8105709157a78) | `cc5f8e867553` | v1.2.1 | Apache-2.0 | P3.2–P3.4 |
| [aws-samples/awsome-inference](https://github.com/aws-samples/awsome-inference/tree/d3236c8cfabb5699c4292f7e0351e4dbc9099de3) | `d3236c8cfabb` | — | MIT | P3.4 |
| [run-ai/runai-model-streamer](https://github.com/run-ai/runai-model-streamer/tree/3c0d220a807770285468aa7d333cb352bac7b1f6) | `3c0d220a8077` | 0.16.1 | Apache-2.0 | P3.7 (#5) |
| [NVIDIA/nccl](https://github.com/NVIDIA/nccl/tree/12df1a11afad322be5a204a2db890161cbf8131d) | `12df1a11afad` | v2.32.3-1 | Apache-2.0 | P4.1 |
| [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests/tree/afd59abab774a6a4b492eb344bd0c845a0b7bb4d) | `afd59abab774` | v2.21.1 | BSD-3-Clause | P4.1 |
| [pytorch/torchtitan](https://github.com/pytorch/torchtitan/tree/a8a66a1907d98a54cc7352f704c2cc2678852944) | `a8a66a1907d9` | v0.3.0 | BSD-3-Clause | P4.2, P4.3 (#9) |
| [aws-samples/awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training/tree/d432521298a4c05afa2077ac1d12dfba69fb0050) | `d432521298a4` | v2.0.1-pre-reorg (pre-reorg tag — use the SHA) | MIT | P4.3, infra/aws |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e) | `5443602d89ed` | v13.4 | BSD-3-Clause | P5.1–P5.3 |
| [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA/tree/5a7dcc513d951ba764d51bc9d587b3163f3a894d) | `5a7dcc513d95` | — | MIT | P5.6, L5 |
| [NVIDIA/cccl](https://github.com/NVIDIA/cccl/tree/547dfed7304c654192e27601cd8e5b2c4b24fcfd) | `547dfed7304c` | v3.5.0 | Apache-2.0 | P5.4, L3 (CUB baseline) |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/0b55a2f691d69981583568fd9eb69687b1f0de8a) | `0b55a2f691d6` | v4.8.0 | BSD-3-Clause | P5.7, L5 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/76d9331028192d06da018c034979765aa0f22899) | `76d933102819` | v2.8.3.post1 | BSD-3-Clause | P5.8, L6 |
| [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal/tree/00f8f46712d493665c900c32fa3a261c8ef3e20c) | `00f8f46712d4` | — | Apache-2.0 | P5.8, L6 |
| [gpu-mode/lectures](https://github.com/gpu-mode/lectures/tree/77a8df418834e5789c12da23e7d2719e0efabef1) | `77a8df418834` | — | Apache-2.0 | P5 |
| [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream/tree/5c0efa14c48f4f53c6c204380dc4bb5a386dae3e) | `5c0efa14c48f` | — | MIT | P5 |
| [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles/tree/b3c4b237d7f0dc6d82055b753c8ea6e0cbb845eb) | `b3c4b237d7f0` | — | MIT | L1–L2 warm-up |
| [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles/tree/4d794abed02292081500ebf4b1e35cd5fc5fb019) | `4d794abed022` | — | Apache-2.0 | P5.9, L6 |
| [triton-lang/triton](https://github.com/triton-lang/triton/tree/4a5147d919defd3064388d4cdd202fb5e9403b61) | `4a5147d919de` | v3.8.0 | MIT | P5.9, L6, #12 |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8) | `bb823b3e0698` | — | MIT | P6 |
| [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges/tree/37a1253e167149376fcc14b4fe745172b23e7646) | `37a1253e1671` | — | CC BY-NC-ND 4.0 | Lane B (titles/mapping only; no statements reproduced) |
| [vvinjamu/silicon-to-scale](https://github.com/vvinjamu/silicon-to-scale/tree/2252f365bb01973148d8b77acb6a4b615c16668c) | `2252f365bb01` | — | MIT | P1–P5 (anchor) |

### 1.1 Path checks against the prompt's starting points

| Prompt said | Status at pinned SHA | Use instead |
|---|---|---|
| llama2.c `run.c` | OK | also `runq.c` (int8 quantized forward pass), a good bridge to D1's int8 dot product |
| llama.cpp `ggml/src/ggml-cpu/` | OK | per-arch SIMD lives in `ggml/src/ggml-cpu/arch/{x86,arm}/quants.c`; generic version in `ggml/src/ggml-cpu/quants.c` |
| llama.cpp `ggml-quants.c` (block formats) | **Moved/split** | block struct definitions (`block_q4_0`, `block_q8_0`, …) are in `ggml/src/ggml-common.h`; reference (de)quantization stays in `ggml/src/ggml-quants.c` |
| nanoGPT `model.py` | OK | — |
| llm.c `train_gpt2.c`, `dev/cuda/` | OK | `dev/cuda/` has 22 standalone kernels, incl. `softmax_forward.cu`, `layernorm_forward.cu`, `attention_forward.cu`, `matmul_forward.cu` |
| HF `models/llama/modeling_llama.py` | OK | — |
| vLLM `benchmarks/` | OK, but | the maintained CLI is `vllm bench …`, implemented in `vllm/benchmarks/`; top-level `benchmarks/` holds micro-benchmarks |
| vLLM `vllm/v1/`, `layers/quantization/`, `v1/spec_decode/` | OK | scheduler: `vllm/v1/core/sched/scheduler.py`; KV: `vllm/v1/core/kv_cache_manager.py`; runner: `vllm/v1/worker/gpu_model_runner.py` |
| vLLM CUDA-graph design doc | OK | `docs/design/cuda_graphs.md`; capture code in `vllm/compilation/cuda_graph.py` |
| vLLM production metrics | OK | `docs/usage/metrics.md` and `docs/design/metrics.md`. Metric names used in the course (e.g. `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_queue_time_seconds`) were read from that file |
| vLLM Run:ai streamer docs | OK | `docs/models/extensions/runai_model_streamer.md` |
| SGLang `python/sglang/srt/`, `mem_cache/radix_cache.py` | OK | scheduler: `python/sglang/srt/managers/scheduler.py` |
| SGLang `sgl-kernel/` | **Moved** | kernels now live in `python/sglang/kernels/` (`jit/`, `aot/`, `ops/`) |
| AutoAWQ | **Deprecated** | README says it is no longer maintained and was adopted by **vLLM LLM Compressor**. Use LLM Compressor for AWQ in #7 |
| CUTLASS CuTe tutorial docs | OK | `media/docs/cpp/cute/` (`00_quickstart.md` … `0x_gemm_tutorial.md`); Python CuTe DSL under `examples/python/CuTeDSL/` |
| nccl-tests `doc/PERFORMANCE.md` | OK | — |
| Triton `python/tutorials/` | OK | 11 tutorials (`01-vector-add.py` … `11-programmatic-dependent-launch.py`) |
| nano-vllm `nanovllm/engine/` | OK | `scheduler.py`, `block_manager.py`, `model_runner.py`, `sequence.py`, `llm_engine.py` |
| flash-attention-minimal | OK | `flash.cu` |
| awsome-distributed-training | OK, but | latest tag is a pre-reorganization snapshot; pin by SHA and re-check paths when building P4 |

## 2. Anchor resources

| Resource | URL | Status | Notes |
|---|---|---|---|
| Modular LLM Inference Handbook | https://handbook.modular.com/ | VERIFIED (via repo) | Site blocked from the build env; source repo pinned above. Page index read from `docs/`. Interactive components present (see §2.1) |
| Inference Engineering Academy | https://www.inference-engineering.xyz/ | **UNVERIFIED** | Site blocked from the build env; no public source repo found. Not used as a dependency until verified |
| AI/ML Infrastructure from Silicon to Scale | https://vvinjamu.github.io/silicon-to-scale/v0.0/index.html | VERIFIED (via repo) | Source repo `vvinjamu/silicon-to-scale` (MIT) holds `index.html`, per-chapter PDFs and 8 diagram pages |

### 2.1 Modular handbook interactives we link instead of rebuilding

| Interactive (component) | Handbook page (repo path → site path) | Used in |
|---|---|---|
| AutoregressiveDecodeStepper, LatencyTimelineVisualizer, ContextWindowSimulator | `docs/llm-inference-basics/how-does-llm-inference-work.md` | P1.2, P1.3 |
| RequestLifecycle | `docs/llm-inference-basics/what-is-llm-inference.md` | P1.1 |
| GPU memory calculator (Calculator) | `docs/getting-started/calculating-gpu-memory-for-llms.md` | P1.5 (D3) |
| GPUTable | `docs/getting-started/choosing-the-right-gpu.md` | P1.4 |
| BatchingSimulator, ChunkedPrefillVisualizer | `docs/inference-optimization/static-dynamic-continuous-batching.md` | P2.3 |
| TopPvsTopK | `docs/model-interaction/inference-parameters.md` | P0.5, P6.5 |
| GPUExecutionVisualizer | `docs/kernel-optimization/gpu-architecture-fundamentals/index.mdx` | P5.1 |
| WarpDivergenceVisualizer | `…/gpu-architecture-fundamentals/threads-warps-blocks.md` | P5.1 |
| SMFloorplan, WarpSchedulerVisualizer | `…/gpu-architecture-fundamentals/streaming-multiprocessors.md` | P1.4, P5.1 |
| MemoryCoalescingVisualizer, MemoryHierarchyExplorer | `…/gpu-architecture-fundamentals/gpu-memory.md` | P5.2 |
| KernelFusionVisualizer | `docs/kernel-optimization/kernel-optimization-for-llm-inference.md` | P5.5 |

Site URLs are the repo path minus `docs/` and the extension (e.g. `https://handbook.modular.com/llm-inference-basics/how-does-llm-inference-work`). **UNVERIFIED** until opened in a browser, because the site was blocked.

### 2.2 Silicon to Scale: reading path vs the prompt

The prompt says the "Inference Fast-Track" is chapters 01, 06, 08, 09, 11, 13, 17. The book's own front matter (`pdfs/ch00_front_matter.pdf`, *Path 2 — The Practicing Inference Engineer*) gives: **Weeks 1–3:** ch 6, 11, 8 · **Weeks 4–6:** ch 7, 9, 17 · **Weeks 7–8:** ch 14, 5. The syllabi follow the book's actual path, add ch 1 (performance mindset) to P1 and ch 13 (speculative/MoE) to P2, so both lists are covered. Chapter files: `ch01_performance_mindset` … `ch18_principal_career`, plus appendices A (hardware reference), B (tools), C (glossary). Chapter numbering also has `ch03a`/`ch03b`.

## 3. LeetGPU

| Item | Status |
|---|---|
| https://leetgpu.com/challenges | **UNVERIFIED** (blocked from the build env) |
| Challenge set | VERIFIED (via repo) `AlphaGPU/leetgpu-challenges@37a1253e`. 107 challenges (19 easy, 70 medium, 18 hard), all `access_tier = "free"` at this SHA. Titles taken from each `challenge.py` `name` field |
| Per-challenge URLs on leetgpu.com | **UNVERIFIED**. The maps link to the challenge folder in the pinned repo (verified) and to the site index |

The prompt's L4 list mentions GELU. The challenge set has no plain GELU; it has **Gaussian Error Gated Linear Unit (GEGLU)**, which we use instead. There is no named FlashAttention challenge; the L6 exit check uses *Softmax Attention* and *Causal Self-Attention*.

## 4. Web documentation, articles and papers

All of these hosts were **blocked by the egress policy** of the environment that built Stage 1, so every entry is **UNVERIFIED** and must be opened before a lesson cites it. Where the docs live in a pinned repo, we cite the repo file instead.

| Source | URL | Fallback verified in a repo |
|---|---|---|
| vLLM docs | https://docs.vllm.ai | `vllm/docs/` at pinned SHA |
| SGLang docs | https://docs.sglang.ai | `sglang/docs/` |
| GPU Operator docs | https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/ | `gpu-operator` repo |
| KEDA Prometheus scaler | https://keda.sh/docs/latest/scalers/prometheus/ | — |
| Karpenter | https://karpenter.sh | — |
| Ray Serve docs | https://docs.ray.io/en/latest/serve/ | `ray/doc/` |
| Triton tutorials (rendered) | https://triton-lang.org/main/getting-started/tutorials/index.html | `triton/python/tutorials/` |
| CUDA C++ Programming / Best Practices Guides | https://docs.nvidia.com/cuda/ | — |
| Nsight Compute / Systems docs | https://docs.nvidia.com/nsight-compute/ · https://docs.nvidia.com/nsight-systems/ | — |
| NVIDIA MIG user guide | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/ | — |
| ERR_NVGPUCTRPERM | https://developer.nvidia.com/ERR_NVGPUCTRPERM | — |
| NVIDIA architecture whitepapers (A100, H100, L4/L40S Ada, Blackwell) | nvidia.com | — **required before any roofline number is printed** |
| Transformer Inference Arithmetic | https://kipp.ly/transformer-inference-arithmetic/ | — |
| siboehm CUDA-MMM article | https://siboehm.com/articles/22/CUDA-MMM | `siboehm/SGEMM_CUDA` |
| Mark Harris reduction PDF | https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf | — |
| Horace He, Go Brrrr | https://horace.io/brrr_intro.html | — |
| HF Ultra-Scale Playbook | https://huggingface.co/spaces/nanotron/ultrascale-playbook | — |
| How to Scale Your Model | https://jax-ml.github.io/scaling-book/ | — |
| PyTorch DCP | https://pytorch.org/docs/stable/distributed.checkpoint.html | — |
| OpenTelemetry Python, Prometheus, Grafana docs | opentelemetry.io · prometheus.io · grafana.com | — |
| Intel Intrinsics Guide · Arm intrinsics · Agner Fog · Brendan Gregg perf/flamegraphs · learncpp | as listed in the prompt | — |
| AWS docs (EC2 types, DLAMI, Service Quotas, Capacity Blocks, SSM, EKS, ParallelCluster, EFA, Budgets) | docs.aws.amazon.com, aws.amazon.com | — **prices must be checked in the learner's region** |
| Second cloud (GKE GPUs, Modal, Lambda, CoreWeave) | as listed | — |
| Google SRE books | https://sre.google/books/ | — |

**Papers** (arXiv blocked; IDs as given in the prompt, UNVERIFIED): PagedAttention 2309.06180 · SGLang 2312.07104 · Orca (OSDI '22) · SARATHI 2308.16369 · Sarathi-Serve 2403.02310 · DistServe 2401.09670 · Speculative decoding 2211.17192 · Medusa 2401.10774 · EAGLE 2401.15077 · AWQ 2306.00978 · GPTQ 2210.17323 · FP8 formats 2209.05433 · FlashAttention 2205.14135 · FA-2 2307.08691 · FA-3 2407.08608.

## 5. Books (summarized and assigned by chapter, never copied; UNVERIFIED editions)

CS:APP (ch 5, 6, 9) · OSTEP (free) · Algorithms for Modern Hardware (free) · C++ Concurrency in Action 2e · PMPP 4e · Designing Data-Intensive Applications · Kubernetes Up & Running · Terraform Up & Running · AI Engineering (Huyen) · Google SRE books.
