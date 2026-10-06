# infra/aws/single-node: one GPU instance for T2 labs

| | |
|---|---|
| Creates | 1 EC2 GPU instance (DLAMI, Ubuntu 22.04), gp3 encrypted root disk, a security group with **no inbound rules**, an IAM role (SSM, CloudWatch agent, read-only S3 prefix, one SSM parameter), an AWS Budget, an idle-stop alarm |
| Access | **SSM Session Manager only.** No SSH key, no open port. Servers bind to `127.0.0.1` and you reach them with `make forward` (an SSM port-forward) |
| Cost | instance $/h (see the table below) + gp3 storage while the disk exists (also when stopped) |
| Teardown | `make down`. `make check` validates the code offline in CI |

## Instance options

> **Prices are not listed here on purpose.** They differ by region and change over time. Look up the on-demand and spot prices for your region on the EC2 pricing page before you `make up`, and write them in your lab notes: every `aws.md` asks for it. GPU specs are in [`gpu_specs.yaml`](../../../course/P1-inference-fundamentals/P1.4-gpu-architecture-and-roofline/gpu_specs.yaml) (UNVERIFIED).

| `instance_type` | GPU | Use in the course |
|---|---|---|
| `g4dn.xlarge` | 1× T4 16 GB | cheapest CUDA: Lane B L1–L4, P5.1–P5.5 |
| `g5.xlarge` | 1× A10G 24 GB | 7–8B fp16 serving (P2.1–P2.4) |
| `g6.xlarge` | 1× L4 24 GB (FP8) | **default**: P2 serving + quantization bakeoff (P2.5), L5 tensor cores |
| `g6e.xlarge` | 1× L40S 48 GB | larger models, longer context |

## One-time setup

1. **Quota.** G and P instance vCPU quotas often start at **0** in new accounts. In the console, open *Service Quotas → Amazon EC2 → "Running On-Demand G and VT instances"* (and the Spot variant if you use spot). Request at least 4 vCPUs (a `g6.xlarge` has 4), or 8 to have headroom. Approval can take from minutes to a few days, so request early.
2. **AWS CLI + Session Manager plugin** on your laptop (`aws --version`, `session-manager-plugin`).
3. **HF token** (for gated models): `aws ssm put-parameter --name /s2s/hf_token --type SecureString --value "$HF_TOKEN"`. Never commit it, and never put it in `terraform.tfvars`.
4. `cp terraform.tfvars.example terraform.tfvars` and fill in your values.

## Every lab

```bash
cd infra/aws/single-node
make up                 # ~2–4 min
make ssm                # shell on the instance
#   on the instance:  curl -fsSL <your fork>/infra/aws/scripts/bootstrap.sh | bash -s -- p2   (or copy it over)
make forward            # in a second terminal: localhost:8000 -> instance 127.0.0.1:8000
make stop               # pause (you still pay for the disk)
make down               # delete everything
```

## Guardrails (all mandatory)

| Guardrail | How |
|---|---|
| idle auto-stop | `s2s-idle-watchdog` (cron, every minute): stops the instance after `idle_minutes` of GPU utilization < 5% with no SSM session |
| backstop | CloudWatch alarm: stop after 3 h of < 2% CPU, in case the watchdog dies |
| budget | AWS Budget: emails at 80% forecast and 100% actual of `monthly_budget_usd` |
| no public endpoint | the SG has no ingress; `make check` fails if anyone adds one; servers bind to 127.0.0.1 |
| secrets | HF token in SSM Parameter Store; the role can read only `/s2s/hf_token` |
| least privilege | the role has SSM core, the CloudWatch agent, and read on one S3 prefix. Nothing else |
| teardown is tested | CI runs `make check` and `terraform plan -destroy` against a mock-credential `plan` (see `tools/ci_terraform.sh`) |

## Drivers and CUDA on the box

```bash
nvidia-smi                       # driver version + GPU
nvcc --version                   # CUDA toolkit (if the AMI ships one; the Base AMI usually does)
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_name())"
```

The driver must be new enough for the CUDA runtime inside your containers (driver forward compatibility; P3.1). Record all three outputs in your lab notes.

## Profiling (P5)

Nsight Compute needs GPU performance-counter access. On this instance you're root via `sudo`, so either run `sudo ncu …`, or allow non-root profiling (see NVIDIA's ERR_NVGPUCTRPERM page):

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf && sudo reboot
```

Copy reports back to your laptop for the desktop UI: `aws s3 cp report.ncu-rep s3://<your-bucket>/` (S3 write needs a policy you add yourself), or `cat` it base64-encoded through the SSM session for small files.
