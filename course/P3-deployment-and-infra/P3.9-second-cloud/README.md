# P3.9: Second cloud (GKE)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for `terraform validate` and the overlay and cost comparisons. ![T3](https://img.shields.io/badge/tier-T3%20cloud%20cluster-red) to deploy on GKE (a GCP project with billing and L4 quota) |
| **Time** | ≈25 min reading + ≈6 h hands-on |
| **Prerequisites** | P3.2–P3.6 (EKS path), P3.5 (#2 cost model) |
| **You will build** | the same platform on a second provider, with only the infra layer and one overlay changed, plus a cross-cloud latency and cost table |

The default is **GKE**, the closest Kubernetes analogue to EKS. The alternatives are below, in case you picked another one at the syllabus review.

## Learning objectives

1. Map each AWS building block to its GCP counterpart, and spot where the semantics differ, not just the names.
2. Keep the platform cloud-agnostic: same manifests, same gateway, same observability, with the cloud confined to `infra/` and one Kustomize overlay.
3. Get GPU quota and plan for it: on a new cloud account the quota process is often the longest step.
4. Compare clouds on **$/1M tokens at the SLO**, not on list price per GPU-hour.

---

## 1. The mapping

| Concern | AWS (P3.2) | GCP (this module) | Difference that bites |
|---|---|---|---|
| Kubernetes | EKS + managed node groups | GKE Standard + node pools | GKE installs the GPU **driver and device plugin** itself (`gpu_driver_installation_config`). No GPU Operator, and DCGM metrics come from GKE's managed DCGM or your own exporter |
| GPU node | `g6.xlarge` (1× L4) | `g2-standard-4` (1× L4) | vCPU/RAM per GPU differ. Re-check your pod requests fit (UNVERIFIED shapes) |
| GPU taint | you add it (`nvidia.com/gpu=present:NoSchedule`) | GKE adds it automatically | same key, so the same toleration works |
| Pod → cloud identity | IRSA / EKS Pod Identity | Workload Identity Federation for GKE | annotate the KSA with a GSA (or bind IAM to the KSA principal directly). The concept is the same, the wiring differs |
| Node metadata protection | IMDSv2 + hop limit 1 | `GKE_METADATA` mode | both stop pods from stealing node credentials |
| Private egress to object storage | S3 gateway endpoint | Private Google Access | weights traffic without NAT charges |
| Egress to the internet | NAT gateway | Cloud NAT | both bill per hour **and** per GB |
| Object storage for weights | S3 | GCS | the P3.7 `Store` protocol needs a `GCSStore`. That's exercise 3's extension |
| Spot | Spot instances (2-min notice) | Spot VMs (preemption notice ~30 s, UNVERIFIED) | a **shorter** notice. Re-size `terminationGracePeriodSeconds`, and expect more mid-stream failures (P3.6) |
| Quota | Service Quotas: vCPU-based "G and VT" | per-GPU-model quotas (`NVIDIA_L4_GPUS`) + `GPUS_ALL_REGIONS` | GCP's global GPU quota starts at 0 on many new projects |
| Budget | AWS Budgets | Cloud Billing budgets | neither stops spending. They only alert |

## 2. What stays the same

Everything above `infra/`: `platform/deploy/base`, the gateway, the observability rules, the autoscaler and the tenancy policies. `overlays/gke/kustomization.yaml` differs from `overlays/eks` **only** in the image registry. `test_overlays.py` enforces that, so cloud-specific drift fails CI. If you notice yourself writing `if cloud == "gcp"` in platform code, stop: it belongs in an overlay or in `infra/`.

## 3. Alternatives to a second Kubernetes cloud

| Option | Less of | More of | Fits when |
|---|---|---|---|
| **Modal** (serverless GPUs) | infra code, cluster ops | per-second pricing at a premium, platform-specific APIs | bursty or batch workloads, small teams |
| **Lambda / CoreWeave** (GPU clouds) | price per GPU-hour | less managed glue, and your own K8s or Slurm | sustained large-GPU training or serving |

Whichever you pick, the deliverable is the same cross-cloud table.

> **Predict first.** Same model, same vLLM flags, both on 1× L4. Should the #4 knee (tokens/s) be the same on EKS and GKE? What would make it differ? (CPU per GPU for tokenization, PCIe topology, driver version.) Which column of the cost table will differ most?

---

## Walkthrough

```bash
bash tools/ci_terraform.sh                                                # T0: validates infra/gcp/gke too
uv run pytest course/P3-deployment-and-infra/P3.9-second-cloud/exercises
cd infra/gcp/gke && cp -n terraform.tfvars.example terraform.tfvars && make up && make creds && make addons
kubectl apply -k platform/deploy/overlays/gke
# same #4 ramp as P3.4/P3.6, through the port-forwarded gateway; then fill the cross-cloud table
```

## What you should see

- `terraform validate`: `Success!` for `infra/gcp/gke` (`TODO(run)`: needs only the terraform binary).
- `test_overlays.py`: the GKE and EKS overlays have the same resources and the same gateway config, and differ only in `images`.
- On GKE (`TODO(run-on: GKE g2-standard-4)`): `kubectl get nodes -L cloud.google.com/gke-accelerator` shows `nvidia-l4` once a GPU pod is pending, and the vLLM pod becomes Ready with no GPU Operator installed.

## Bench: cross-cloud table

| cloud | node | $/GPU-h (your region, date) | #4 knee tok/s | p90 TTFT @ knee | $/1M tokens @ 100% | @ 30% util | cold start s |
|---|---|---|---|---|---|---|---|
| AWS | g6.xlarge | | `TODO(run-on: EKS)` | | | | |
| GCP | g2-standard-4 | | `TODO(run-on: GKE)` | | | | |

`compare_clouds.py` fills the cost columns from your prices and #4 results.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Terraform for a GKE GPU node pool, `validate` / `plan` in CI](exercises/01-terraform.md) | T0 | `ci_terraform.sh` + `test_gke_tf.py` (static policy) |
| 2 | [Same manifests, only an overlay changes](exercises/02-overlays.md) | T0 | `test_overlays.py` |
| 3 | [Cross-cloud latency and cost table](exercises/03-compare.md) | T0 (with your numbers) / T3 | `test_compare.py` + your filled table |

## Common mistakes

- Assuming the GPU quota exists. On GCP, request `GPUS_ALL_REGIONS` **and** the per-model regional quota days ahead.
- Installing the GPU Operator on GKE alongside GKE's managed driver.
- Comparing list $/GPU-h instead of $/1M tokens at the SLO.
- Copying the EKS spot grace period onto GKE Spot VMs and their shorter notice.
- Leaving the cluster up: the system pool and NAT bill while the GPU pool is at zero.

## Go deeper

- GKE docs: *Run GPUs in GKE Standard node pools*, Workload Identity Federation, Spot VMs (all UNVERIFIED).
- Modal, Lambda and CoreWeave docs (UNVERIFIED).
- Modular handbook: *Multi-cloud and cross-region inference*, *Bring your own cloud*.

**Next:** [P4 Multi-GPU and distributed](../../P4-multi-gpu-and-distributed/syllabus.md).
