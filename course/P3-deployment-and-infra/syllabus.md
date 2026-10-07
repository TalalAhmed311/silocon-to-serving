# P3 — Deployment and infra on multiple clouds

**Weeks 13–20 · Lane A ≈ 88 h · Tier T3 (AWS EKS GPU nodes) with T0 parts (kind cluster on a laptop for every manifest, Terraform `plan`/`validate`, unit tests) · Lane B: L2 → L3 (from week 15)**

**Thread through the phase:** turn P2's single-node server into a platform. Package it, run it on Kubernetes with GPUs, observe it, autoscale it on queue depth, load weights quickly, and isolate tenants, on AWS plus one second cloud. Projects #1, #13, #2, #3, #5 and #8 are all layers of `platform/`.

Pinned: GPU Operator `v26.7.1`, DCGM exporter `4.8.4`, k8s-device-plugin `v0.20.1`, KServe `v0.21.0`, Dynamo `v1.5.0`, llm-d `v0.10.0`, Ray `2.59.0`, KEDA `v2.21.0`, terraform-aws-eks `v21.26.0`, data-on-eks `v1.2.1`, Run:ai Model Streamer `0.16.1`.

**Cost rule for this phase:** every lab starts with `make up` and ends with `make down`. `make down` is checked locally with `terraform plan -destroy` before each lab. Budget alarm + idle auto-stop are created by the same Terraform. Spend estimates are computed at build time from in-region prices (TODO(verify-prices)).

| Module | Time | Project | Animations |
|---|---|---|---|
| P3.1 GPU containers | 0.4 h + 7 h | (images used by #1) | — |
| P3.2 Terraform on AWS | 0.5 h + 10 h | (`infra/aws/` single-node + EKS) | — |
| P3.3 Kubernetes + NVIDIA GPU Operator | 0.5 h + 9 h | — | — |
| P3.4 Serving stacks on K8s | 0.6 h + 14 h | **#1 self-hosted inference cluster** | **request path gateway → queue → autoscaler → pod, cold vs warm** (shared with P3.6) |
| P3.5 Observability | 0.6 h + 12 h | **#13 observability spine**, **#2 cost-per-token dashboard** | — (Mermaid diagrams) |
| P3.6 Autoscaling, cold starts, spot | 0.5 h + 10 h | **#3 queue-based GPU autoscaler** | same as P3.4 |
| P3.7 Weight storage and lazy loading | 0.4 h + 8 h | **#5 model weight delivery** | — |
| P3.8 Multi-tenancy and security | 0.4 h + 7 h | **#8 secure multi-tenant layer** (stretch) | — |
| P3.9 Second cloud | 0.4 h + 7 h | (mirror of single-node + K8s) | — |

---

## P3.1 — GPU containers

**Objectives.** (1) The layers: host driver ↔ container CUDA toolkit/runtime compatibility (forward-compat rules from NVIDIA docs, cited). (2) Build slim serving images with multi-stage builds, pinned base digests and no model weights baked in. (3) NVIDIA Container Toolkit and `--gpus`. (4) Image size and pull time, and why they matter for cold start (P3.6).

**Examples.** `env/Dockerfile.cpu`, `env/Dockerfile.cuda-dev`, `env/Dockerfile.serving` (pinned vLLM) · `01_check_cuda_compat.sh` (prints `nvidia-smi` driver version and the container's `nvcc --version` / torch CUDA).

**Exercises.** (1) Hadolint-clean Dockerfile (CI check) · (2) cut an image by ≥30% and show `docker image ls` before/after · (3) run-as-non-root with a read-only root FS · (4) *(hard)* reproducible build: same digest twice.

**Sources.** NVIDIA Container Toolkit and CUDA compatibility docs (UNVERIFIED) · vLLM `docker/Dockerfile` and `docs/deployment/` at the pinned SHA.

## P3.2 — Terraform on AWS

**Objectives.** (1) Terraform state, modules, workspaces, and remote state in S3 with locking. (2) **Single-node module:** one GPU instance (DLAMI), SSM-only access (no port 22), gp3/NVMe for weights, an IAM role with read-only access to one S3 prefix, a budget alarm, and an idle auto-stop (CloudWatch alarm on GPU util from the CloudWatch agent's NVIDIA metrics, or a cron watchdog). (3) **EKS module:** VPC + `terraform-aws-eks` with a GPU managed node group (or Karpenter NodePool), spot + on-demand fallback. (4) Quotas: G/P vCPU quotas often start at 0, so you request them through Service Quotas.

**Examples.** `infra/aws/single-node/` · `infra/aws/eks/` · `infra/aws/guardrails/` (budget, auto-stop) · `Makefile` with `up`, `down`, `ssm`, `port-forward`.

**Exercises.** (1) `terraform validate` + `tflint` + `checkov` clean (T0, CI) · (2) add an S3 VPC endpoint so weight pulls skip the NAT · (3) prove there is no inbound 0.0.0.0/0 rule (a policy test) · (4) *(hard)* auto-stop fires after N idle minutes. The test drives it with synthetic metrics.

**Sources.** terraform-aws-eks `examples/` · data-on-eks GPU inference blueprints · AWS docs: DLAMI, SSM Session Manager, Service Quotas, Budgets, EKS GPU AMIs (UNVERIFIED) · *Terraform: Up & Running*.

## P3.3 — Kubernetes + NVIDIA GPU Operator

**Objectives.** (1) The K8s pieces you need: Deployments, Services, probes, resources/limits, taints/tolerations, node affinity, PDBs. (2) GPU Operator components: driver, container toolkit, device plugin, GFD labels, DCGM exporter, MIG manager. (3) Schedule `nvidia.com/gpu: 1`. Readiness that waits for the model to load. Graceful drain of in-flight streams.

**Examples.** `01_kind_cpu/` (T0: the same manifests with a CPU mock backend) · `02_gpu_operator_values.yaml` · `03_vllm_deployment.yaml` (startupProbe sized from the measured load time).

**Exercises.** (1) Manifests pass `kubeconform` + `kube-score` (T0) · (2) readiness gates on `/health` after warm-up · (3) `preStop` drain test on kind · (4) *(hard)* node-feature labels: schedule only on GPUs with ≥24 GB.

**Sources.** gpu-operator repo + docs · k8s-device-plugin · *Kubernetes Up & Running* · EKS user guide GPU sections.

## P3.4 — Serving stacks on K8s (+ #1)

**Objectives.** (1) Compare KServe (InferenceService, a vLLM runtime), NVIDIA Dynamo (disaggregated prefill/decode, KV-aware routing), llm-d (K8s-native distributed inference, inference-scheduler) and Ray Serve (Python-first composition). (2) Prefill–decode disaggregation and when it pays (DistServe). (3) **#1:** a multi-node GPU cluster serving an open model behind #6. Health checks, a reproducible public latency report from #4, and no unauthenticated public endpoint.

**Examples.** `01_kserve_isvc.yaml` · `02_dynamo/` (following its pinned deploy guide) · `03_llmd/` · `04_rayserve_app.py`.

**Exercises.** (1) Same model on two stacks, one comparison table (#4 harness) · (2) rolling update with zero dropped requests · (3) **#1 cluster** checklist (see `platform/README.md`) · (4) *(hard)* enable disaggregated serving on Dynamo or llm-d and measure TTFT/ITL vs aggregated at the same GPU count.

**Animation (required).** `animations/p3-request-path.html`: request → gateway → queue → (autoscaler scales from 0/1) → pod. Toggle cold vs warm to see where the seconds go: node provision, image pull, weight load, graph capture.

**Sources.** KServe, Dynamo, llm-d, Ray Serve (pinned) · awsome-inference · DistServe (2401.09670) · handbook `prefill-decode-disaggregation.md`, `distributed-inference.md` · Inference Engineering Academy fleet/Dynamo course (UNVERIFIED).

## P3.5 — Observability (+ #13, #2)

**Objectives.** (1) The three signals: metrics, traces, logs. SLIs/SLOs and error budgets. (2) Scrape vLLM metrics (names verified from `docs/usage/metrics.md` and `vllm/v1/metrics/` at the pinned SHA, e.g. `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:kv_cache_usage_perc`, `vllm:num_requests_waiting`) and DCGM metrics (names from `etc/default-counters.csv` in the pinned dcgm-exporter). (3) OpenTelemetry traces across gateway → backend with spans for queue/prefill/decode and cache hits. (4) **#13:** dashboards + alerts (SLO burn rate, drift in output length, cost spike). (5) **#2:** $/1M tokens by model, tenant and route = (node $/h × GPU share) ÷ tokens/h.

**Examples.** `01_prom_stack/` (kube-prometheus-stack values) · `02_otel_gateway.py` · `03_grafana_dashboards/*.json` · `04_cost_model.py` (T0).

**Exercises.** (1) PromQL for p90 TTFT from histograms (tested with promtool unit tests, T0) · (2) multi-window burn-rate alert rules (promtool tests) · (3) trace propagation test across gateway and mock backend (T0) · (4) cost model with spot vs on-demand and MIG share (pytest) · (5) *(hard)* a cost-spike alert that fires on a synthetic tenant flood.

**Sources.** Google SRE books (SLO and alerting chapters) · OpenTelemetry Python, Prometheus, Grafana docs · dcgm-exporter · handbook `comprehensive-observability.md`, `build-and-maintenance-cost.md` · Silicon to Scale ch 17.

## P3.6 — Autoscaling, cold starts, spot (+ #3)

**Objectives.** (1) Why CPU/GPU-util HPA is wrong for LLMs and queue depth (`vllm:num_requests_waiting`) or a concurrency target is right. (2) KEDA ScaledObject with the Prometheus scaler, and scale-to-zero trade-offs. (3) Cold-start anatomy and mitigations: warm pools, pre-pulled images, node over-provisioning, fast weight loading (P3.7). (4) Spot interruption handling (two-minute notice, drain) with on-demand fallback via Karpenter or node-group priorities.

**Exercises.** (1) The scaling formula as a pure function with tests (T0) · (2) KEDA ScaledObject on kind with the mock backend (T0) · (3) **#3** on EKS: a load ramp from #4 and a scale-up timeline · (4) *(hard)* simulated spot interruption with zero failed requests.

**Bench.** `cold start phase | seconds` (provision, pull, load, warm-up) before and after mitigations. `TODO(run-on: EKS g6.xlarge nodes)`.

**Sources.** KEDA + Prometheus scaler docs · Karpenter docs · handbook `fast-scaling.md`.

## P3.7 — Weight storage and lazy loading (+ #5)

**Objectives.** (1) Where load time goes: S3 → network → disk → page cache → host → GPU. Reuse P0.2's mmap and pinned-memory lessons. (2) safetensors sharding and index files. (3) Streaming loaders: Run:ai Model Streamer with vLLM `--load-format runai_streamer`, checked against `docs/models/extensions/runai_model_streamer.md`. (4) **#5:** a tiny weight registry (versioned S3 prefixes + manifest + checksums), a node-local NVMe cache with an LRU, and a loader benchmark.

**Exercises.** (1) Manifest + checksum verifier (T0) · (2) LRU NVMe cache with size limit (T0) · (3) S3 throughput vs concurrency curve (T2/T3) · (4) *(hard)* stream shards straight to GPU and compare with download-then-load.

**Bench.** `method | model GB | load s | effective GB/s`.

**Sources.** safetensors · runai-model-streamer · vLLM model loading docs · Silicon to Scale ch 15.

## P3.8 — Multi-tenancy and security (+ #8, stretch)

**Objectives.** (1) Tenancy models: shared pool vs dedicated deployments vs MIG slices (P4.4). (2) Keyed access, per-tenant quotas (from #6), NetworkPolicies, namespaces, Pod Security Standards. (3) Audit logs and secrets handling (Secrets Manager/SSM, HF token at runtime, IRSA/Pod Identity least privilege). (4) A sandboxed tool/exec path (gVisor or Firecracker overview; T0 demo with a seccomp profile).

**Exercises.** (1) NetworkPolicy tests on kind · (2) key rotation without downtime · (3) tamper-evident audit log (hash chain, pytest) · (4) *(hard)* noisy-neighbor test: one tenant cannot break another's SLO.

**Sources.** K8s docs (UNVERIFIED) · AWS Secrets Manager/IAM docs · handbook `inferenceops-and-management.md`.

## P3.9 — Second cloud

**Objectives.** Mirror the single-node and K8s paths on **one** second provider and call out the differences: identity, networking, GPU node pools, quota process, pricing model. Default: **GKE** with GPU node pools (the closest K8s analogue). Alternative: Modal (serverless, less infra code) or a GPU cloud (Lambda/CoreWeave). The choice is confirmed with the learner at the syllabus review.

**Exercises.** (1) Terraform for a GKE GPU node pool, `plan` only in CI · (2) the same Helm values deploy on both clouds with only an overlay changing · (3) cross-cloud latency/cost table using #2.

**Sources.** GKE GPUs how-to · Modal / Lambda / CoreWeave docs (all UNVERIFIED) · handbook `multi-cloud-and-cross-region-inference.md`, `bring-your-own-cloud.md`.
