# GAPS

This file lists everything marked `UNVERIFIED` or `TODO(run-on: …)`, with the exact step that closes it. It was regenerated at the end of Stage 4 with `python tools/list_gaps.py`.

## A. Hosts blocked from the Stage 1 build environment

The build sandbox's egress policy denied these hosts. Re-run `python tools/check_links.py --external` from a machine with normal internet access to close them.

| Host / source | Why it matters | How to close |
|---|---|---|
| leetgpu.com | per-problem URLs in all six `leetgpu-map.md` files | open https://leetgpu.com/challenges in a browser; confirm the titles match `AlphaGPU/leetgpu-challenges@37a1253` and record the URL pattern; then regenerate the maps |
| www.inference-engineering.xyz | anchor resource (P1 foundations, P2 engines, P3 fleet/Dynamo) | open it in a JS-capable browser; record the course/lesson index and pin the access date; until then no lesson depends on it |
| handbook.modular.com | anchor; content verified through the repo, but the site URLs are not | spot-check 5 page URLs and the interactives listed in SOURCES.md §2.1 |
| vvinjamu.github.io | anchor; verified through the repo | open the index and the diagrams page once |
| arxiv.org, usenix.org | 15 paper citations | open each ID in SOURCES.md §4 and confirm title/authors |
| docs.nvidia.com, nvidia.com | **whitepaper numbers for every roofline** (P1.4) | download the A100, H100, Ada (L4/L40S) and Blackwell whitepapers and the T4/A10G datasheets; record page numbers in `gpu_specs.yaml` |
| docs.aws.amazon.com, aws.amazon.com | instance specs and **prices** | check each instance in §5 (`g4dn.xlarge`, `g5.xlarge`, `g6.xlarge`, `g6e`, a 4×L4 `g6.12xlarge`-class instance, `p4d.24xlarge`, `p5`) in the learner's region |
| docs.vllm.ai, docs.sglang.ai, docs.ray.io, keda.sh, karpenter.sh, triton-lang.org | rendered docs | the repo `docs/` folders at the pinned SHAs were used instead; spot-check the rendered pages |
| kipp.ly, siboehm.com, horace.io, en.algorithmica.org, huggingface.co/spaces, jax-ml.github.io, sre.google, opentelemetry.io, prometheus.io, grafana.com, pytorch.org, intel.com, arm.com, agner.org, brendangregg.com, learncpp.com, csapp.cs.cmu.edu, pages.cs.wisc.edu, cloud.google.com, modal.com, docs.lambda.ai, docs.coreweave.com | articles, books, docs | open each link once |

## B. Decisions and facts

Decided (Stage 2 approval, "go with your defaults"):

- Silicon to Scale order: the book's Path 2 (6, 11, 8 → 7, 9, 17 → 14, 5), not the prompt's chapter list.
- P3.9 second cloud: **GKE** (`infra/gcp/gke`).
- P4.3 primary path: **Ray on EKS** (`platform/training/ray`), with Slurm documented (`platform/training/slurm`).

Still to check:

- The P2 default 7–8B model and the P6.7 benchmark model: check each licence and pin by HF revision SHA before the first T2 run.
- `awsome-distributed-training` paths used in P4.3 at the pinned tag.
- Every vLLM / SGLang / nano-vllm file path cited in P6 ("verify at the pin" notes): files move between releases.
- `results/` is git-ignored. For the C2 teardown, commit the specific result files it cites (`git add -f results/<file>.json`) so its links resolve.

## C. Nothing has been executed yet

The course was written without running any code: no tests, no examples, no benchmarks, no Terraform. Everything was checked by review, Python `ast` parsing, YAML parsing and `tools/check_links.py` only. So the **first** gap to close is T0:

```bash
uv sync && S2S_SOLUTIONS=1 uv run pytest                  # every T0 test, exercises run against their solutions
uv run --extra torch pytest -m torch                      # PyTorch-on-CPU tests
python tools/check_links.py && python tools/build_site.py # docs
bash tools/ci_terraform.sh                                # fmt + validate every stack (no apply)
```

Expect some failures on first run; fix them in the module, not the test, unless the test is wrong.

## D. Runs that need hardware (`TODO(run-on: …)`)

Each entry is a module or folder and the hardware its TODO markers name; the exact commands are in that module's `aws.md` (or `README.md`). "T0 (not yet executed)" is a `TODO(run)` that needs no GPU.

| Where | Runs needed (hardware × count) |
|---|---|
| `course/P0-systems-primer/P0.1-c-cpp-for-systems` | T0 (not yet executed) × 1 |
| `course/P0-systems-primer/P0.2-virtual-memory-and-mmap` | T0 (not yet executed) × 1 |
| `course/P0-systems-primer/P0.3-threads-atomics-caches` | T0 (not yet executed) × 1 |
| `course/P0-systems-primer/P0.4-simd-and-roofline` | T0 (not yet executed) × 1 |
| `course/P0-systems-primer/P0.5-project-tiny-engine-cpu` | T0 (not yet executed) × 1 |
| `course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache` | T0 (not yet executed) × 1 |
| `course/P1-inference-fundamentals/P1.3-latency-and-throughput-metrics` | T0 (not yet executed) × 1 |
| `course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline` | g6.xlarge × 3 |
| `course/P1-inference-fundamentals/P1.5-project-capacity-calculator` | T0 (not yet executed) × 2, g6.xlarge × 2 |
| `course/P1-inference-fundamentals/syllabus.md` | g6.xlarge × 2 |
| `course/P2-serving-engines/P2.1-first-serve-with-vllm` | g6.xlarge × 1 |
| `course/P2-serving-engines/P2.2-sglang` | g6.xlarge × 1 |
| `course/P2-serving-engines/P2.3-batching-and-caching` | g6.xlarge × 1 |
| `course/P2-serving-engines/P2.4-benchmarking-methodology` | T0 (not yet executed) × 1 |
| `course/P2-serving-engines/P2.5-quantization` | g6.xlarge × 1 |
| `course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs` | g6.xlarge × 1 |
| `course/P2-serving-engines/P2.7-gateway-basics` | T0 (not yet executed) × 1 |
| `course/P2-serving-engines/syllabus.md` | g6.xlarge × 2 |
| `course/P3-deployment-and-infra/P3.1-gpu-containers` | g4dn.xlarge × 1 |
| `course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator` | EKS g6.xlarge × 1 |
| `course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s` | EKS 2× g6.xlarge × 1 |
| `course/P3-deployment-and-infra/P3.5-observability` | EKS × 1 |
| `course/P3-deployment-and-infra/P3.6-autoscaling` | EKS + Karpenter × 1, EKS g6.xlarge × 5 |
| `course/P3-deployment-and-infra/P3.7-weight-delivery` | g6.xlarge × 7 |
| `course/P3-deployment-and-infra/P3.9-second-cloud` | EKS × 1, EKS + GKE × 1, GCP × 1, GKE × 2, GKE g2-standard-4 × 1, T0 (not yet executed) × 2 |
| `course/P3-deployment-and-infra/syllabus.md` | EKS g6.xlarge nodes × 1 |
| `course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives` | g6.12xlarge × 1, g6.12xlarge, p4d.24xlarge × 3, p4d.24xlarge × 1 |
| `course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies` | g6.12xlarge × 4 |
| `course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing` | EKS g6.12xlarge × 2 |
| `course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing` | p4d.24xlarge × 3 |
| `course/P4-multi-gpu-and-distributed/syllabus.md` | p4d.24xlarge × 2 |
| `course/P5-cuda-deep-dive/P5.1-execution-model` | g4dn.xlarge × 2 |
| `course/P5-cuda-deep-dive/P5.10-custom-op-into-serving` | g6.xlarge × 3 |
| `course/P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory` | g4dn.xlarge × 3 |
| `course/P5-cuda-deep-dive/P5.3-profiling-with-nsight` | g4dn.xlarge × 2 |
| `course/P5-cuda-deep-dive/P5.4-reductions-and-scans` | g4dn.xlarge × 3 |
| `course/P5-cuda-deep-dive/P5.5-softmax-and-norms` | g4dn.xlarge × 3 |
| `course/P5-cuda-deep-dive/P5.6-gemm-ladder` | g4dn.xlarge × 3 |
| `course/P5-cuda-deep-dive/P5.7-tensor-cores` | g6.xlarge × 4 |
| `course/P5-cuda-deep-dive/P5.8-flashattention` | g4dn.xlarge × 2, g6.xlarge × 1 |
| `course/P5-cuda-deep-dive/P5.9-triton` | g6.xlarge × 2 |
| `course/P6-engine-internals/P6.1-reading-nano-vllm` | g6.xlarge × 2 |
| `course/P6-engine-internals/P6.2-scheduler` | g6.xlarge × 1 |
| `course/P6-engine-internals/P6.3-paged-kv-block-manager` | g6.xlarge × 2 |
| `course/P6-engine-internals/P6.5-spec-decode-and-gpu-sampling` | g6.xlarge × 4 |
| `course/P6-engine-internals/P6.6-cuda-graph-capture` | g6.xlarge × 5 |
| `course/P6-engine-internals/P6.7-project-tiny-engine-gpu` | g6.xlarge × 3 |
| `course/capstone/C1-multi-region-failover` | AWS, two regions × 2 |
| `course/capstone/README.md` | AWS, two regions × 1, T0 (not yet executed) × 2 |
| `infra/aws` | AWS, two regions × 1 |
| `laneB-cuda/L2-memory-and-shared-memory/syllabus.md` | g4dn.xlarge × 1 |
| `laneB-cuda/L3-reductions-and-scans/solutions` | g4dn.xlarge × 2 |
| `laneB-cuda/L3-reductions-and-scans/syllabus.md` | g4dn.xlarge × 1 |
| `laneB-cuda/L4-fused-elementwise-and-norms/solutions` | g4dn.xlarge × 1 |
| `laneB-cuda/L5-gemm-and-tensor-cores/syllabus.md` | g6.xlarge × 1 |
| `laneB-cuda/L6-attention-and-triton/syllabus.md` | g6.xlarge × 1 |
| `platform/engine` | T0 (not yet executed) × 1, T2 × 5, g6.xlarge × 2 |
| `platform/failover` | EKS, two regions × 1 |
| `platform/kernels` | T0 (not yet executed) × 1 |
| `platform/partitioning` | p4d.24xlarge × 1 |

## E. `UNVERIFIED` markers

Spec-sheet numbers, prices, flags and doc paths quoted without a primary source open at build time. Close each by citing the source (URL + date or page) next to the number and removing the marker.

| Where | UNVERIFIED markers |
|---|---|
| `README.md` | 1 |
| `SOURCES.md` | 9 |
| `course/P1-inference-fundamentals/P1.1-transformer-forward-pass` | 3 |
| `course/P1-inference-fundamentals/P1.2-prefill-decode-kv-cache` | 3 |
| `course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline` | 24 |
| `course/P1-inference-fundamentals/P1.5-project-capacity-calculator` | 3 |
| `course/P1-inference-fundamentals/syllabus.md` | 2 |
| `course/P2-serving-engines/P2.1-first-serve-with-vllm` | 2 |
| `course/P2-serving-engines/P2.2-sglang` | 2 |
| `course/P2-serving-engines/P2.4-benchmarking-methodology` | 1 |
| `course/P2-serving-engines/P2.5-quantization` | 1 |
| `course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs` | 2 |
| `course/P2-serving-engines/syllabus.md` | 1 |
| `course/P3-deployment-and-infra/P3.1-gpu-containers` | 2 |
| `course/P3-deployment-and-infra/P3.2-terraform-on-aws` | 2 |
| `course/P3-deployment-and-infra/P3.3-kubernetes-and-gpu-operator` | 3 |
| `course/P3-deployment-and-infra/P3.4-serving-stacks-on-k8s` | 1 |
| `course/P3-deployment-and-infra/P3.6-autoscaling` | 3 |
| `course/P3-deployment-and-infra/P3.7-weight-delivery` | 1 |
| `course/P3-deployment-and-infra/P3.8-multi-tenancy-and-security` | 2 |
| `course/P3-deployment-and-infra/P3.9-second-cloud` | 5 |
| `course/P3-deployment-and-infra/syllabus.md` | 5 |
| `course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives` | 4 |
| `course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies` | 1 |
| `course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing` | 4 |
| `course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing` | 6 |
| `course/P5-cuda-deep-dive/P5.1-execution-model` | 1 |
| `course/P5-cuda-deep-dive/P5.10-custom-op-into-serving` | 2 |
| `course/P5-cuda-deep-dive/P5.2-memory-coalescing-and-shared-memory` | 2 |
| `course/P5-cuda-deep-dive/P5.6-gemm-ladder` | 1 |
| `course/P5-cuda-deep-dive/P5.7-tensor-cores` | 3 |
| `course/P5-cuda-deep-dive/aws-common.md` | 1 |
| `course/P5-cuda-deep-dive/syllabus.md` | 1 |
| `course/P6-engine-internals/P6.1-reading-nano-vllm` | 1 |
| `course/capstone/C1-multi-region-failover` | 1 |
| `course/capstone/syllabus.md` | 1 |
| `infra/aws` | 11 |
| `infra/gcp` | 5 |
| `laneB-cuda/L1-launch-and-indexing/leetgpu-map.md` | 1 |
| `laneB-cuda/L2-memory-and-shared-memory/leetgpu-map.md` | 1 |
| `laneB-cuda/L3-reductions-and-scans/leetgpu-map.md` | 1 |
| `laneB-cuda/L4-fused-elementwise-and-norms/leetgpu-map.md` | 1 |
| `laneB-cuda/L5-gemm-and-tensor-cores/leetgpu-map.md` | 1 |
| `laneB-cuda/L6-attention-and-triton/leetgpu-map.md` | 1 |
| `platform/autoscaler` | 2 |
| `platform/bakeoff` | 3 |
| `platform/capacity` | 3 |
| `platform/deploy` | 5 |
| `platform/kernels` | 3 |
| `platform/observability` | 1 |
| `platform/partitioning` | 6 |
| `platform/tenancy` | 2 |
| `platform/training` | 4 |
| `platform/weights` | 1 |
| `reports` | 1 |
