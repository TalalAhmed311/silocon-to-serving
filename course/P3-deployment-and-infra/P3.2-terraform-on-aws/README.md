# P3.2 — Terraform on AWS

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for writing, validating and policy-testing all the code; ![T3](https://img.shields.io/badge/tier-T3%20AWS%20cluster-red) to apply the EKS stack. The single-node stack is T2 |
| **Time** | ≈30 min reading + ≈10 h hands-on (≈2 h of EKS uptime) |
| **Prerequisites** | P2.1 (you've used `infra/aws/single-node`) |
| **Pinned** | `terraform-aws-modules/eks` **v21.26.0**, Terraform ≥ 1.6 |

## Learning objectives

1. Terraform fundamentals: providers, resources, modules, state (remote in S3, with locking), `plan`/`apply`/`destroy`.
2. Read and extend `infra/aws/single-node`: SSM-only access, IMDSv2, an encrypted disk, least-privilege IAM, a budget, an idle auto-stop with a backstop alarm.
3. Build `infra/aws/eks`: a VPC, EKS with a tainted scale-from-zero GPU node group, and an S3 gateway endpoint.
4. Handle quotas and costs as first-class requirements.

## Why this matters

Clicking around the console is how cloud bills explode and environments become unreproducible. Every T2/T3 lab in the course is one `make up` / `make down` away, and you can read and review each guardrail as code.

---

## 1. Terraform in one page

```hcl
provider "aws" { region = "us-east-1" }            # which API, which credentials (from your environment)
resource "aws_instance" "gpu" { ... }               # a thing that exists
module "vpc" { source = "terraform-aws-modules/vpc/aws"  version = "6.0.1" ... }   # a reusable bundle, PINNED
output "id" { value = aws_instance.gpu.id }
```

- **State** maps the code to real resource IDs. Keep it remote, in an S3 backend with locking (S3 native locking or a DynamoDB table, depending on your Terraform version), so two people or laptops never apply at once. **Never commit `terraform.tfstate`**: it can contain secrets.
- **Plan, then apply.** Read the plan. "1 to destroy" on a database is a bad day.
- **Pin every module version** and provider range. `terraform init -upgrade` is a deliberate act.

## 2. The single-node stack, guardrail by guardrail

Read `infra/aws/single-node/*.tf` with this table next to it:

| File | Guardrail | What to look for |
|---|---|---|
| `main.tf` | no inbound | the security group has **only** egress |
| `main.tf` | IMDSv2 | `http_tokens = "required"`: SSRF in a served app can't steal instance credentials |
| `main.tf` | stop, not terminate | `instance_initiated_shutdown_behavior = "stop"`: the watchdog keeps your disk |
| `iam.tf` | least privilege | SSM core, the CloudWatch agent, S3 read on **one prefix**, one SSM parameter |
| `guardrails.tf` | budget + backstop | an AWS Budget, and a CloudWatch alarm with the `ec2:stop` action after 3 h idle |
| `user_data.sh.tftpl` | idle auto-stop | a cron watchdog: GPU util < 5% and no SSM session for N minutes → `shutdown` |

## 3. The EKS stack

`infra/aws/eks` composes three pinned modules:

- **VPC**: 2 AZs, private subnets for nodes, public subnets for load balancers, one NAT (a cost compromise, noted).
- **EKS v21.26.0** with two managed node groups:
  - `system` (1× m6i.large) for cluster add-ons
  - `gpu` (`g6.xlarge` by default) with the `AL2023_x86_64_NVIDIA` AMI type, so drivers are preinstalled. It is tainted `nvidia.com/gpu=present:NoSchedule`, so only pods that tolerate the taint land there, and has `min_size = 0`, so you pay for GPUs only while something needs them. P3.6's autoscaler drives it.
- A **guardrails** module (Budget).

The EKS API endpoint is public but **restricted to your IP** (`allowed_api_cidrs`). `make check` refuses `0.0.0.0/0`.

## 4. Quotas, again

New accounts often have a G/VT on-demand quota of **0 vCPUs**. EKS adds nothing to that, but the GPU node group can't scale up without it. Request it before P3 (Service Quotas → EC2 → "Running On-Demand G and VT instances"; for P4, the "P instances" quota too). Approval can take days.

> **Predict first.** Your EKS lab runs 6 hours: the control plane, a NAT gateway, 1× m6i.large, and a `g6.xlarge` for 3 of those hours. Look up the five in-region prices and compute the total before you `make up`. Then run `infra/aws/scripts/cost.sh` the next day and compare (Cost Explorer lags by up to 24 h). Which line item surprised you? (For most people: the NAT gateway's per-hour charge, and its per-GB processing.)

---

## Walkthrough

```bash
bash tools/ci_terraform.sh                                    # T0: fmt + validate every stack, ingress policy
uv run pytest course/P3-deployment-and-infra/P3.2-terraform-on-aws/exercises
cd infra/aws/eks && cp -n terraform.tfvars.example terraform.tfvars && make up && make addons   # T3
kubectl get nodes -L s2s/pool
make down
```

## What you should see

- `ci_terraform.sh` ends with `terraform checks OK`, or, at first, the module input names that need fixing for v21.26.0. That is expected, because the names are UNVERIFIED (see the EKS README).
- On T3: one `system` node; the GPU pool shows no nodes until a pod requests `nvidia.com/gpu` (P3.3).

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Static checks: fmt, validate, tflint, checkov](exercises/01-static-checks.md) | T0 | `tools/ci_terraform.sh` clean; checkov findings triaged in writing |
| 2 | [The S3 gateway endpoint: why, and prove it](exercises/02-s3-endpoint.md) | T0 + T3 | a route-table check in the plan; NAT bytes before/after |
| 3 | [Policy test: no inbound, no public API](exercises/03-policy.md) | T0 | `test_policy_tf.py` |
| 4 | [Idle auto-stop, tested with synthetic metrics](exercises/04-idle-stop.md) | T0 | `test_idle.py` |

## Common mistakes

- Local state, or a committed `terraform.tfstate`.
- `0.0.0.0/0` on anything inbound "for a minute".
- Forgetting that a NAT gateway and the EKS control plane cost money while the GPU pool is at 0.
- `terraform destroy` hanging because K8s created LoadBalancers and ENIs that Terraform doesn't know about. `make down` deletes those first.

## Go deeper

- `terraform-aws-modules/terraform-aws-eks@e0724620` (v21.26.0): `examples/` and the README's input table. That table is the source of truth for exercise 1.
- `awslabs/data-on-eks@cc5f8e86`: GPU inference blueprints (Terraform + add-ons).
- AWS docs: EKS user guide (GPU AMIs, managed node groups), Session Manager, Service Quotas, AWS Budgets. UNVERIFIED from the build environment.
- *Terraform: Up & Running*: state, modules, workflows.

**Next:** [P3.3 Kubernetes + NVIDIA GPU Operator](../P3.3-kubernetes-and-gpu-operator/README.md).
