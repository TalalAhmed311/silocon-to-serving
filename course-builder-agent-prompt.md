# Agent Prompt: Build the "Silicon-to-Serving" Course

You are a senior AI infrastructure engineer, a CUDA performance engineer and a technical course designer, working as one agent. Your job is to turn the curriculum outline below into a **complete, runnable, self-paced course**. The course must include:

- lessons
- worked code examples
- interactive animations
- exercises with tests
- benchmarks
- deployment guides

The learner is one person. They want to become a full-fledged **AI infra and inference engineer** who can deploy and operate LLM serving on cloud GPUs and write CUDA C++ kernels.

Work in the repository you are given (or create `silicon-to-serving/`). Everything you produce must run either on a laptop, on a single local NVIDIA GPU, in the LeetGPU browser playground, or on an AWS GPU instance. Each example is clearly labelled with which one it needs.

---

## 0. Non-negotiable rules

1. **Verify before you cite.** Open every URL, repo and file path before you reference it. Repository paths in this prompt are starting points from mid-2026; repos get refactored. If a path moved, find the new one. If you cannot verify something, mark it `UNVERIFIED` and do not build on it.
2. **Pin versions.** Every external repo you point to is referenced by commit SHA or release tag. Every Python environment has a lockfile (`uv.lock` or `requirements.txt` with pins). Every CUDA example states the minimum CUDA version and compute capability.
3. **No invented numbers.** Every benchmark figure in a lesson comes either from code in this repo that you ran, with the hardware stated, or from a cited source. If you could not run it (no GPU available), write the expected *shape* of the result and leave a `TODO(run-on: g6.xlarge)` marker with the exact command.
4. **Respect copyright and licenses.** Link to and summarize external material in your own words; do not paste chapters, blog posts or docs. Do not vendor third-party code into the course. Reference it by SHA and explain it. Short excerpts (≤15 lines) from permissively licensed code are fine for annotation, with the license and source noted.
5. **Every code example runs.** It has a one-line run command, a correctness test against a reference implementation (NumPy, PyTorch or cuBLAS), and, where performance is the point, a benchmark script that prints a table.
6. **Teach by predicting, then measuring.** Every performance lesson asks the learner to predict a number (GB/s, FLOP/s, tokens/s, $ per 1M tokens) from first principles before running the code, then compare.
7. **Cost and safety on cloud.** Every AWS guide includes:
   - the instance type and hourly cost range
   - a teardown command
   - an auto-shutdown safeguard
   - no unauthenticated public endpoints
8. **Checkpoint with the human.** Stop for review after Stage 2 (the syllabus) and after the first fully built module (P0.1). Do not build the whole course before the first review.

---

## 1. The curriculum you are implementing

The structure is: a short **systems primer (P0)** first, then a **top-down** path from serving and deployment down to CUDA kernels and engine internals.

Two lanes run in parallel:

- **Lane A (phases + projects):** about 11 h/week.
- **Lane B (CUDA kernels on LeetGPU):**
  - 3–4 problems a week during P0–P4.
  - Daily from P5 onward.

The schedule assumes about 15 h/week and roughly 44 weeks.

### Phases (Lane A)

| Phase | Weeks | Topic | Core content |
|---|---|---|---|
| P0 | 1–3 | Systems primer: C/C++, OS, memory, SIMD | C pointers and memory layout; C++17/20 (RAII, move, templates, CMake); processes vs threads; virtual memory, page tables, TLB, page faults; `mmap`, page cache, pinned vs pageable memory, DMA; `std::thread`, atomics, false sharing, thread pools; caches, NUMA; intro roofline; AVX2/AVX-512/NEON intrinsics; `perf` |
| P1 | 4–6 | How LLM inference works + hardware math | Transformer forward pass (attention, MLP, RMSNorm, RoPE); prefill vs decode; KV cache sizing; TTFT, ITL/TPOT, throughput, goodput; GPU roofline (HBM bandwidth vs tensor-core FLOPs); SM/warp/HBM/L2 overview |
| P2 | 7–12 | Serving with inference engines | vLLM and SGLang; continuous batching, chunked prefill, prefix caching; quantization (FP8, INT8, AWQ, GPTQ); speculative decoding; CUDA graphs; TP flags; benchmarking methodology; gateway basics |
| P3 | 13–20 | Deployment and infra on multiple clouds | GPU Docker images; Kubernetes + NVIDIA GPU Operator; KServe / NVIDIA Dynamo / llm-d / Ray Serve; Terraform on AWS + one other cloud; OpenTelemetry, Prometheus, DCGM exporter, Grafana, SLOs; KEDA autoscaling, cold starts, spot; weight storage and lazy loading; multi-tenancy |
| P4 | 21–24 | Multi-GPU, GPU sharing, distributed jobs | NCCL collectives; TP/PP/EP/DP, FSDP; NVLink/PCIe/EFA bandwidth math; MIG, time-slicing, MPS; Ray and Slurm jobs, checkpointing, fault tolerance |
| P5 | 25–34 | CUDA C++ deep dive | Execution model, occupancy, divergence; coalescing, shared memory, bank conflicts; Nsight Compute and Nsight Systems; GEMM optimization ladder; tensor cores (WMMA → `mma.sync` → CuTe); Triton; wiring a custom op into PyTorch/vLLM |
| P6 | 35–40 | Engine internals | Scheduler (continuous batching, chunked prefill, preemption); PagedAttention block manager; prefix cache (radix tree); spec-decode verification; CUDA graph capture; GPU sampling |
| Cap | 41–44 | Capstone | Multi-region failover; public reproducible benchmark teardown |

### Lane B: CUDA C++ levels mapped to LeetGPU (https://leetgpu.com/challenges)

| Level | Weeks | Concepts | Problem types | Exit check |
|---|---|---|---|---|
| L1 | 1–6 | Launch, 1D/2D indexing, bounds, copies, error checks | vector add, ReLU/leaky ReLU, color inversion, reverse array, matrix copy, naive matmul | write 2D indexing for any shape unaided |
| L2 | 7–14 | Coalescing, shared memory, `__syncthreads`, bank conflicts, atomics, occupancy | transpose, 1D/2D convolution, histogram, count elements, tiled matmul | transpose ≥80% of copy bandwidth |
| L3 | 15–24 | Tree reduction, warp shuffles, scan, cooperative groups | reduction, dot product, prefix sum, sort, top-k, SpMV | reduction within 10% of `cub::DeviceReduce` |
| L4 | 25–27 | Online softmax, fused norms, `float4` loads, fp16/bf16 | softmax, LayerNorm/RMSNorm, GELU/SwiGLU, batch norm, cross-entropy | single-pass online softmax, reported in GB/s |
| L5 | 28–31 | GEMM ladder, tensor cores, async copies, double buffering | matmul, FP16 GEMM, INT8 quantized matmul, batched GEMM | SGEMM ≥70% cuBLAS; tensor-core HGEMM ≥50% |
| L6 | 32–44 | Tiled attention, FlashAttention v1→v2, paged/decode attention, Triton rewrites | softmax attention, MHA, causal attention, hard leaderboard problems | FA-2 forward ≥5× naive at seq 4k |

LeetGPU's challenge list renders client-side. Use a browser tool to read the real titles and map each challenge to a level. Do not reproduce problem statements; link to them and write your own hint ladder and reference solution outline.

### GEMM ladder (L5) — the rungs every learner implements

1. naive
2. coalesced
3. shared-memory tiling
4. 1D register blocking
5. 2D register blocking
6. vectorized loads + transposed A tile
7. warp tiling + autotuning + double buffering
8. tensor cores

### Projects (portfolio, built as layers of ONE platform repo)

| ID | Phase | Tier | Project | Build |
|---|---|---|---|---|
| #0 v0 | P0 | core | Tiny Inference Engine (CPU) | llama2.c-style Llama forward pass in C++: mmap'd safetensors, SIMD matmul, thread pool, KV cache, top-p sampling |
| D1 | P0 | drill | SIMD SGEMM + roofline | naive → tiled → AVX/NEON → multithreaded, roofline plot; int8 dot product |
| D2 | P0 | drill | KV block allocator | fixed-size block allocator, free list, refcounts, stress test (precursor to PagedAttention) |
| D3 | P1 | drill | Capacity calculator | predicts VRAM, KV size, max batch, decode tokens/s per model×GPU |
| #4 | P2 | core | Continuous batching load test | traffic generator to KV-cache saturation; find the latency knee |
| #7 | P2 | core | Quantized serving bakeoff | FP16 vs AWQ vs FP8: quality vs latency vs VRAM |
| #10 | P2 | strong | Speculative decoding prototype | draft + target, prefix caching, chunked prefill; acceptance rate and real speedup |
| #6 | P2 | strong | Multi-model AI gateway | routing, retries, fallbacks, rate limits, per-tenant token budgets across 2–3 backends |
| #1 | P3 | core | Self-hosted inference cluster | multi-node GPU K8s cluster serving an open model, health checks, public latency report |
| #13 | P3 | core | Observability spine | traces for queue/prefill/decode/cache hits/errors; drift and cost-spike alerts |
| #2 | P3 | core | Cost-per-token dashboard | TTFT, ITL, DCGM GPU util, $/1M tokens by model, tenant, route |
| #3 | P3 | core | Queue-based GPU autoscaler | KEDA on queue depth, cold-start mitigation, spot fallback |
| #5 | P3 | strong | Model weight delivery | safetensors registry, sharding, lazy loading / streaming, NVMe cache |
| #8 | P3 | stretch | Secure multi-tenant layer | isolation, keyed access, sandboxed tool/exec paths, audit logs |
| #9 | P4 | strong | Checkpointed distributed training | Ray or Slurm + FSDP/TP, fault injection, resume from checkpoint |
| #11 | P4 | stretch | GPU partitioning lab | MIG / time-slicing / MPS, fair scheduling, contention tests |
| D4 | P5 | drill | Kernels repo | reduction, softmax, RMSNorm, GEMM ladder, int8 matmul, FA-2 forward with benchmarks |
| #12 | P5 | core | Custom kernel path | replace one hot path in the serving stack with own Triton/CUDA kernel; before/after via #13 |
| #0 v1 | P6 | core | Tiny Inference Engine (GPU) | port #0 to CUDA with continuous batching + paged KV (D2) + D4 kernels; plug into #6; benchmark vs vLLM with #4 |
| #14 | Cap | strong | Multi-region failover drill | active-passive/active-active across regions/clouds, DNS failover, RTO/RPO notes |
| #15 | Cap | core | Public benchmark teardown | reproducible architecture + latency/cost report of the whole platform |

How the platform stacks, top to bottom:

- clients and tenants (load from #4)
- gateway (#6)
- tenancy (#8)
- autoscaler (#3)
- serving backends (vLLM/SGLang, #7, #10, #12, #0)
- GPU nodes, with MIG (#11) and weights from #5

Running alongside every layer:

- observability (#13)
- cost tracking (#2)

Around the whole stack:

- failover (#14) wraps everything
- the teardown (#15) documents it

---

## 2. Where to look: source map

Read these to build each module. Before using any repo, open the link, confirm the file path still exists, pin the commit SHA, and note the license in `SOURCES.md`. File paths inside repos are starting points and may have moved.

### The learner's three anchor resources (use these as the spine)

- **Modular LLM Inference Handbook** — https://handbook.modular.com/
  - Full page index: https://handbook.modular.com/llms.txt. Append `.md` to any page URL to get Markdown.
  - Source repo: https://github.com/modular/llm-inference-handbook
  - Use it for P1–P5 concepts.
  - Link its interactive tools (GPU memory calculator, KV cache calculator, batching simulator, coalescing visualizer) from your lessons instead of rebuilding them.
- **Inference Engineering Academy** — https://www.inference-engineering.xyz/
  - Start page: https://www.inference-engineering.xyz/course/foundations/inference-path
  - Hands-on serving with vLLM, SGLang and Dynamo.
  - Foundations → P1, engine courses → P2, fleet/Dynamo → P3.
  - The site needs JavaScript; read it with a browser tool.
- **AI/ML Infrastructure from Silicon to Scale** (Venkat Vinjam) — https://vvinjamu.github.io/silicon-to-scale/v0.0/index.html
  - Diagrams: https://vvinjamu.github.io/silicon-to-scale/v0.0/index.html#diagrams
  - Follow its "Inference Fast-Track" chapters (01, 06, 08, 09, 11, 13, 17) across P1–P5.
  - Chapters 10–15 → P4. Chapter 18 → interview prep.

### P0 — systems primer

**Books** (summarize and assign chapters; do not copy):

- *Computer Systems: A Programmer's Perspective* (Bryant & O'Hallaron), ch. 5, 6, 9 — https://csapp.cs.cmu.edu/
- *Operating Systems: Three Easy Pieces*, free — https://pages.cs.wisc.edu/~remzi/OSTEP/ (virtualization + concurrency parts)
- *Algorithms for Modern Hardware*, free — https://en.algorithmica.org/hpc/ (SIMD: https://en.algorithmica.org/hpc/simd/ · matmul: https://en.algorithmica.org/hpc/algorithms/matmul/)
- *C++ Concurrency in Action*, 2nd ed. (Anthony Williams)

**Repos and files:**

- llama2.c — https://github.com/karpathy/llama2.c
  - `run.c` is the reference shape for #0 v0: weight mapping, forward pass, sampling.
  - Its README lists C++ and SIMD ports.
- llama.cpp / ggml — https://github.com/ggml-org/llama.cpp
  - `ggml/src/ggml-cpu/`: real AVX2 / AVX-512 / NEON kernels and quantized dot products.
  - Find where the block quantization formats are defined now; it used to be `ggml-quants.c`.
- safetensors — https://github.com/huggingface/safetensors
  - File format spec, for the mmap loader.

**References:**

- Intel Intrinsics Guide — https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
- Arm NEON intrinsics (for Apple silicon learners) — https://developer.arm.com/architectures/instruction-sets/intrinsics/
- Agner Fog's optimization manuals — https://www.agner.org/optimize/
- Linux `perf` and flame graphs — https://www.brendangregg.com/perf.html and https://www.brendangregg.com/flamegraphs.html
- learncpp — https://www.learncpp.com

### P1 — inference fundamentals

- Transformer Inference Arithmetic (kipply) — https://kipp.ly/transformer-inference-arithmetic/
- nanoGPT — https://github.com/karpathy/nanoGPT (`model.py`)
- llm.c — https://github.com/karpathy/llm.c
  - `train_gpt2.c` for the CPU forward pass.
  - `dev/cuda/` for standalone kernels.
- Hugging Face Llama implementation — https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
  - Covers RoPE, attention and KV cache handling.
- NVIDIA architecture whitepapers (Ampere A100, Hopper H100, Ada L4/L40S, Blackwell) — search nvidia.com for "<GPU> architecture whitepaper".
  - These are the source for bandwidth and FLOPs numbers in roofline math. Cite the exact document.

### P2 — serving engines

**vLLM:**

- Repo: https://github.com/vllm-project/vllm
- Docs: https://docs.vllm.ai
- Files to read:
  - `benchmarks/` for benchmark scripts
  - `vllm/v1/` for engine core, scheduler and KV cache manager
  - `vllm/model_executor/layers/quantization/` for quantization
  - `vllm/v1/spec_decode/` for speculative decoding
- Verify the current layout of all of these.

**SGLang:**

- Repo: https://github.com/sgl-project/sglang
- Docs: https://docs.sglang.ai
- Files to read:
  - `python/sglang/srt/` for the scheduler, and `mem_cache/radix_cache.py` for the prefix cache
  - `sgl-kernel/` for kernels

**Load testing:**

- GuideLLM (load generation, SLO benchmarking) — https://github.com/vllm-project/guidellm

**Quantization (#7):**

- LLM Compressor — https://github.com/vllm-project/llm-compressor
- AutoAWQ — https://github.com/casper-hansen/AutoAWQ (check whether it is archived and which tool replaced it)
- GPTQModel — https://github.com/ModelCloud/GPTQModel
- Quality scoring with lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness

**Gateways (#6)** — read their routing and rate-limit code, then build your own:

- LiteLLM — https://github.com/BerriAI/litellm
- Envoy AI Gateway — https://github.com/envoyproxy/ai-gateway

**Papers** (link and summarize):

- PagedAttention / vLLM — https://arxiv.org/abs/2309.06180
- SGLang / RadixAttention — https://arxiv.org/abs/2312.07104
- Orca (continuous batching) — https://www.usenix.org/conference/osdi22/presentation/yu
- SARATHI (chunked prefill) — https://arxiv.org/abs/2308.16369 · Sarathi-Serve — https://arxiv.org/abs/2403.02310
- DistServe (prefill–decode disaggregation) — https://arxiv.org/abs/2401.09670
- Speculative decoding (Leviathan et al.) — https://arxiv.org/abs/2211.17192
- Medusa — https://arxiv.org/abs/2401.10774 · EAGLE — https://arxiv.org/abs/2401.15077
- AWQ — https://arxiv.org/abs/2306.00978 · GPTQ — https://arxiv.org/abs/2210.17323
- FP8 formats — https://arxiv.org/abs/2209.05433

### P3 — deployment and infra

**Kubernetes and GPUs:**

- NVIDIA GPU Operator — https://github.com/NVIDIA/gpu-operator · docs: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/
- DCGM exporter — https://github.com/NVIDIA/dcgm-exporter
- Kubernetes device plugin — https://github.com/NVIDIA/k8s-device-plugin

**Serving stacks:**

- KServe — https://github.com/kserve/kserve
- NVIDIA Dynamo — https://github.com/ai-dynamo/dynamo
- llm-d — https://github.com/llm-d/llm-d
- Ray Serve — https://github.com/ray-project/ray · docs: https://docs.ray.io/en/latest/serve/

**Autoscaling (#3):**

- KEDA — https://github.com/kedacore/keda
- KEDA Prometheus scaler — https://keda.sh/docs/latest/scalers/prometheus/
- Karpenter — https://karpenter.sh

**AWS infra code:**

- EKS Terraform module — https://github.com/terraform-aws-modules/terraform-aws-eks
- Data on EKS (GPU inference blueprints) — https://github.com/awslabs/data-on-eks
- awsome-inference — https://github.com/aws-samples/awsome-inference

**Weight delivery (#5):**

- Run:ai Model Streamer — https://github.com/run-ai/runai-model-streamer
- vLLM's model loading docs at https://docs.vllm.ai

**Observability (#13, #2):**

- OpenTelemetry Python — https://opentelemetry.io/docs/languages/python/
- Prometheus — https://prometheus.io/docs/
- Grafana — https://grafana.com/docs/
- vLLM metrics: look up the production metrics page in the vLLM docs and use those metric names.

**Books** (summarize, assign chapters):

- Google SRE books, free — https://sre.google/books/
- *Designing Data-Intensive Applications* (Kleppmann)
- *Kubernetes Up & Running*
- *Terraform: Up & Running*
- *AI Engineering* (Chip Huyen)

### P4 — multi-GPU and distributed

**NCCL:**

- NCCL — https://github.com/NVIDIA/nccl
- nccl-tests — https://github.com/NVIDIA/nccl-tests
  - Run `all_reduce_perf`.
  - Explain bus bandwidth vs algorithm bandwidth, using `doc/PERFORMANCE.md` in that repo.

**Scaling guides:**

- Hugging Face Ultra-Scale Playbook — https://huggingface.co/spaces/nanotron/ultrascale-playbook
- How to Scale Your Model — https://jax-ml.github.io/scaling-book/

**Distributed training (#9):**

- torchtitan (FSDP2, TP, checkpointing) — https://github.com/pytorch/torchtitan
- PyTorch Distributed Checkpoint — https://pytorch.org/docs/stable/distributed.checkpoint.html
- Slurm / EKS with EFA on P-instances — https://github.com/aws-samples/awsome-distributed-training

**GPU partitioning (#11):**

- NVIDIA MIG user guide — https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- GPU Operator MIG docs, under the GPU Operator docs link above.

### P5 — CUDA deep dive

**Books and lectures:**

- *Programming Massively Parallel Processors*, 4th ed. (Hwu, Kirk, El Hajj) as the spine. The authors' lectures are on YouTube; search "Izzat El Hajj PMPP".

**NVIDIA docs:**

- CUDA C++ Programming Guide — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA C++ Best Practices Guide — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Nsight Compute — https://docs.nvidia.com/nsight-compute/
- Nsight Systems — https://docs.nvidia.com/nsight-systems/

**Repos and articles:**

- CUDA samples — https://github.com/NVIDIA/cuda-samples
- GEMM ladder:
  - Repo — https://github.com/siboehm/SGEMM_CUDA
  - Article — https://siboehm.com/articles/22/CUDA-MMM
- Reduction:
  - Mark Harris, "Optimizing Parallel Reduction in CUDA" — https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  - Baselines from CUB/Thrust in CCCL — https://github.com/NVIDIA/cccl
- CUTLASS / CuTe — https://github.com/NVIDIA/cutlass
  - `examples/`
  - The CuTe tutorial docs folder; verify the current path.
- FlashAttention:
  - Production — https://github.com/Dao-AILab/flash-attention
  - Teaching version (≈100 lines) — https://github.com/tspeterkim/flash-attention-minimal
  - Papers: FA-1 https://arxiv.org/abs/2205.14135 · FA-2 https://arxiv.org/abs/2307.08691 · FA-3 https://arxiv.org/abs/2407.08608
- GPU MODE:
  - Lectures — https://github.com/gpu-mode/lectures
  - Resource stream — https://github.com/gpu-mode/resource-stream
- Puzzles:
  - GPU Puzzles — https://github.com/srush/GPU-Puzzles
  - Triton Puzzles — https://github.com/srush/Triton-Puzzles
- Triton:
  - Tutorials in the repo — https://github.com/triton-lang/triton/tree/main/python/tutorials
  - Rendered tutorials — https://triton-lang.org/main/getting-started/tutorials/index.html
- Making Deep Learning Go Brrrr (Horace He) — https://horace.io/brrr_intro.html
- LeetGPU challenges — https://leetgpu.com/challenges

### P6 — engine internals

- nano-vllm — https://github.com/GeeeekExplorer/nano-vllm
  - Read first: the scheduler, block manager and model runner under `nanovllm/engine/`.
  - It is small enough to annotate line by line.
- Then map each nano-vllm concept to vLLM `vllm/v1/` and SGLang `python/sglang/srt/` (links above).
- vLLM CUDA graphs: find the design doc page in https://docs.vllm.ai and the capture code in the repo.

### AWS and second-cloud references (for §5)

**EC2 and setup:**

- EC2 accelerated instance types — https://aws.amazon.com/ec2/instance-types/ (verify GPU, memory and price per type and region)
- AWS Deep Learning AMIs — https://docs.aws.amazon.com/dlami/
- Service Quotas, for requesting G/P vCPU quota — https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html
- EC2 Capacity Blocks for ML — https://aws.amazon.com/ec2/capacityblocks/
- Session Manager, for access with no open SSH port — https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html

**Clusters:**

- EKS user guide (GPU and ML sections) — https://docs.aws.amazon.com/eks/latest/userguide/
- AWS ParallelCluster — https://docs.aws.amazon.com/parallelcluster/
- Elastic Fabric Adapter — https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html

**Cost and profiling:**

- AWS Budgets — https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html
- Nsight GPU counter permission fix — https://developer.nvidia.com/ERR_NVGPUCTRPERM

**Second cloud:**

- GKE with GPUs — https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
- Modal — https://modal.com/docs
- Lambda — https://docs.lambda.ai
- CoreWeave — https://docs.coreweave.com

---

## 3. What to produce

### 3.1 Repository layout

```
silicon-to-serving/
  README.md                  # how to use the course, hardware tiers, quickstart
  SOURCES.md                 # every external source: URL, SHA/version, license, which modules use it
  site/                      # MkDocs Material site (mkdocs.yml) that renders the whole course
  env/                       # Dockerfiles: cpu, cuda-dev, serving; uv/pip lockfiles
  infra/aws/                 # Terraform + scripts for every AWS environment used (see §5)
  course/
    P0-systems-primer/
      P0.1-c-cpp-for-systems/
      P0.2-virtual-memory-and-mmap/
      P0.3-threads-atomics-caches/
      P0.4-simd-and-roofline/
      P0.5-project-tiny-engine-cpu/
    P1-.../ ... P6-.../  capstone/
  laneB-cuda/
    L1-.../ ... L6-.../       # each with leetgpu-map.md, solutions/, bench/
  platform/                   # the single portfolio repo skeleton the projects grow into
  animations/                 # source for every animation, plus built HTML
  tools/                      # link checker, bench runner, notebook executor
```

### 3.2 Module template (every module folder contains)

- `README.md` — the lesson:
  - learning objectives
  - prerequisites
  - hardware tier badge
  - time estimate
  - concept explanation in your own words with diagrams
  - "Predict first" box
  - walkthrough of the code
  - "What you should see" with real measured output
  - common mistakes
  - "Go deeper" links to sections of the anchor resources and books
  - "Go down when…" pointers to lower-level modules
- `examples/` — numbered, runnable examples, smallest first.
  - Each has a header comment with purpose, run command, expected output and hardware.
  - C/C++/CUDA examples build with CMake or a Makefile.
  - Python examples have pinned deps.
  - Where it adds understanding, provide the same idea twice: naive and optimized, or CPU and GPU.
- `exercises/` — 3–6 exercises, graded easy → hard.
  - Each has a starter file, a hidden-reference solution in `solutions/`, and a test (`pytest`, `ctest`, or a CUDA test harness comparing against a reference with tolerances).
- `bench/` — a script that runs the benchmark and prints a Markdown table (`problem size | time | GB/s or TFLOP/s | % of peak`).
  - It also saves JSON so the site can chart it.
- `animations/` — links to the animations this module uses (see §4).
- `notebook.ipynb` (optional) — for Python-heavy modules, executed end to end in CI on CPU.
- `quiz.md` — 5–10 conceptual questions with answers and explanations.
- `aws.md` — only if the module needs cloud GPUs: instance, cost, launch, run, teardown.

### 3.3 Lane B (LeetGPU) folders

For each level:

- `leetgpu-map.md` — table of real LeetGPU challenges (title, link, difficulty, concept tested).
- A local harness so every kernel can also run and be tested on a local GPU or AWS.
- An annotated solution per problem, with a short "why this is fast" note.
- Profiler screenshots or `ncu` CSV summaries from a real run where available.

---

## 4. Animations and visuals

Animations are a first-class deliverable, not decoration.

**Tooling:**

- Default to self-contained HTML + vanilla JS + SVG/Canvas, one file per animation, no network dependencies, with play/pause/step controls and a speed slider. Embed them in the MkDocs site.
- Use Mermaid for static flow and architecture diagrams.
- Optionally provide Manim scenes for short videos.
- Each animation must work in light and dark mode and at phone width.

**Required animations (minimum set):**

| Module | Animation |
|---|---|
| P0 | virtual address → page table → TLB → physical page; page fault on first touch of an mmap'd file |
| P0 | false sharing between two cores on one cache line |
| P0 | SIMD lanes: scalar loop vs 8-wide AVX2 add; why misalignment costs |
| P0/P1 | roofline: drag arithmetic intensity, watch a kernel move from memory- to compute-bound |
| P1 | prefill vs decode token flow with a growing KV cache; TTFT/ITL timeline |
| P2 | static vs continuous batching timeline; chunked prefill letting decodes continue |
| P2 | PagedAttention: logical blocks → physical blocks, sharing on prefix hit, copy-on-write |
| P2 | speculative decoding: draft proposes k tokens, target verifies, accept/reject |
| P3 | request path through gateway → queue → autoscaler → pod, cold start vs warm |
| P4 | ring all-reduce step by step; TP split of an MLP across 2/4 GPUs |
| P5 | grid → blocks → warps → SM scheduling; warp divergence |
| P5 | memory coalescing: contiguous vs strided access, sectors fetched |
| P5 | shared-memory tiling of matmul; bank conflicts with and without padding |
| P5 | tree reduction vs warp-shuffle reduction |
| P5 | online softmax (running max and sum) |
| P5 | FlashAttention tiling over Q/K/V blocks |
| P6 | scheduler loop: waiting / running / preempted queues per step |
| P6 | radix-tree prefix cache insert / match / evict |

Where the Modular handbook already has an excellent interactive (e.g. its coalescing visualizer or batching simulator), link to it and build only what it lacks.

---

## 5. Hardware tiers and AWS deployment

Label every example and module with one tier:

| Tier | Where | Used for |
|---|---|---|
| T0 | Laptop, no GPU (Linux/macOS, x86 or ARM) | P0, P1 calculators, gateway logic, unit tests, Triton interpreter mode where possible |
| T1 | LeetGPU in the browser | Lane B L1–L4 |
| T2 | One local or rented NVIDIA GPU | P2 single-node serving, P5 kernels + Nsight |
| T3 | AWS multi-GPU / multi-node | P3 cluster, P4 distributed, MIG, failover |

**AWS guidance to bake into `infra/aws/` and every `aws.md`:**

- **Instance mapping.** Verify current availability and prices in the learner's region before writing them down. Starting points:

  | Instance | GPU | Use |
  |---|---|---|
  | g4dn.xlarge | T4 16 GB, no bf16/FP8 | cheapest CUDA practice |
  | g5.xlarge | A10G 24 GB | 7–8B fp16 serving |
  | g6.xlarge | L4 24 GB, FP8 support | quantization bakeoff |
  | g6e | L40S 48 GB | larger models |
  | p4d.24xlarge | 8×A100 + NVLink + EFA | MIG, TP/PP |
  | p5 | 8×H100 | FP8, TP/PP at scale |
  | Capacity Blocks for ML | — | short multi-GPU sessions |

- **AMI and drivers.** Use the AWS Deep Learning AMI (GPU, Ubuntu) or the EKS-optimized accelerated AMI. Document the CUDA driver/toolkit versions and how to check them (`nvidia-smi`, `nvcc --version`).
- **Quotas.** Explain that G and P instance vCPU quotas often start at 0. Give the Service Quotas request steps and expected wait.
- **Single-node path (T2).** Provide:
  - Terraform (or a short CLI script) to launch one GPU instance with SSM Session Manager access (no open SSH port), an attached gp3 or local NVMe volume for weights, and an IAM role for S3 reads.
  - A `bootstrap.sh` that installs the course env, pulls a pinned model, and runs the module's bench.
- **Cluster path (T3).** EKS with a GPU node group via `terraform-aws-eks` + GPU Operator + DCGM exporter + Prometheus/Grafana + KEDA. Document Karpenter or managed node groups with spot + on-demand fallback for #3.
- **Distributed path (P4).** ParallelCluster/Slurm or EKS with EFA for p4d/p5, following `aws-samples/awsome-distributed-training`. Include `nccl-tests` with expected bus bandwidth ranges, cited.
- **Profiling on cloud.** Nsight Compute needs GPU performance-counter access. Document running with `sudo` or setting `NVreg_RestrictProfilingToAdminUsers=0`, and pulling `.ncu-rep` files back for the desktop UI.
- **Cost guardrails** (mandatory in every guide):
  - estimated $/hour
  - an AWS Budget alarm
  - an idle auto-stop (cron or CloudWatch alarm on GPU util)
  - spot where safe
  - a `make down` / `terraform destroy` teardown that is tested
- **Security.**
  - No public inference endpoint without auth.
  - Secrets go in AWS Secrets Manager or SSM Parameter Store, never in the repo.
  - Least-privilege IAM.
  - The HF token is passed at runtime.
- **Second cloud** (for P3 multi-cloud and #14). Mirror the single-node and K8s path on one of GCP (GKE with GPU node pools) or a GPU cloud (CoreWeave, Lambda, Modal), with the differences called out.

---

## 6. Working process

### Stage 1 — Source index

1. Verify every source in §2. Record URL, SHA or version, license and status in `SOURCES.md`.
2. Use a browser tool to pull the real LeetGPU challenge list and map it to L1–L6.
3. Report anything missing, moved or paywalled.

### Stage 2 — Syllabus (STOP for human review)

For every phase and Lane B level, write `syllabus.md`:

- module list with objectives
- example list
- exercise list
- animations used
- tier
- time estimate
- which anchor-resource sections and repo files it draws on

**Stop and ask for approval.**

### Stage 3 — Pilot module (STOP for human review)

1. Build P0.1 end to end with the full module template, one animation, exercises with tests, bench output, and the site page.
2. Run everything.
3. **Stop and ask for feedback on depth, tone and format.**

### Stage 4 — Build out

Build in course order: P0 → P1 → P2 → P3 → P4 → P5 → P6 → Capstone.

- Build Lane B levels alongside, at the weeks they are scheduled.
- Grow the `platform/` repo as projects are reached.

After each phase:

- run all T0 code in CI
- execute notebooks
- link-check
- update `SOURCES.md`
- write a short phase report: what was built, what was run on which hardware, open TODOs

### Stage 5 — QA and delivery

Run the following:

- full link check
- all tests
- every T0 example
- every T2/T3 example you have hardware for

Then:

- Build the site.
- Produce `PROGRESS.md`, a checklist the learner ticks off week by week.
- Produce `GAPS.md`: everything marked `UNVERIFIED` or `TODO(run-on: …)`, with the exact command to finish it.

---

## 7. Quality bar

- **Code-first teaching.** A lesson that has no code the learner runs or writes is incomplete.
- **Readable code.** Every non-obvious line in example code has a comment explaining *why*, not *what*.
- **Lesson length.** About 15–40 minutes of reading per module, then hands-on time. Split anything longer.
- **Plain language.** Introduce each term once with a one-line definition. Keep a `GLOSSARY.md`.
- **One worked prediction per module.** Each module includes at least one prediction worked with real numbers. Example: "Llama-3-8B fp16 on an L4, batch 1 decode: weights ≈16 GB, HBM ≈300 GB/s → ceiling ≈18 tokens/s; measure and explain the gap." Verify every figure you use.
- **Rigorous tests.** Tests compare against references with stated tolerances (fp16/bf16 tolerances explained). Benchmarks warm up, repeat, and report median and p90.
- **Lane B is CUDA C++ first.** Triton versions come as a second pass in L6 and in #12.

Begin with Stage 1.
