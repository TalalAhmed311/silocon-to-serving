# infra/gcp/gke: the P3.9 second cloud

A zonal GKE cluster that mirrors [`infra/aws/eks`](../../aws/eks/README.md): private nodes, a 2-node system pool, and a **min-0 L4 GPU pool** (`g2-standard-4`). It adds Workload Identity, Cloud NAT and a billing budget.

| | |
|---|---|
| Cost | GKE cluster management fee (one zonal cluster may be covered by the free tier, UNVERIFIED) + 2 × `e2-standard-4` + **GPU node-hours** + NAT. Look all of them up in your region: `____ $/h` |
| Quota | `NVIDIA_L4_GPUS` (regional) ≥ `gpu_max`, and `GPUS_ALL_REGIONS` ≥ `gpu_max`. Request it in the console under IAM & Admin → Quotas. Approval can take days |
| Safety | private nodes (no public IPs), the control plane open only to `master_authorized_cidrs`, Workload Identity metadata so pods can't read node credentials, Services stay ClusterIP |

```bash
cp -n terraform.tfvars.example terraform.tfvars && make check
make up && make creds && make addons
kubectl apply -k ../../../platform/deploy/overlays/gke
make down
```

**Auto-stop:** the GPU pool scales to 0 when no GPU pod is pending. The system pool, NAT and cluster fee bill until `make down`, and the billing budget alerts at 50/80/100%. Budgets **alert**. They don't stop spending.

Field names follow the `hashicorp/google` provider docs (UNVERIFIED for the version you pin). `bash tools/ci_terraform.sh` runs `terraform validate` on this directory without credentials.
