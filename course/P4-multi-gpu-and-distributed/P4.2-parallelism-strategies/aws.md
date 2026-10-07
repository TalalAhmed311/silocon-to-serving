# P4.2 on AWS

| | |
|---|---|
| Instance | `g6.12xlarge` (4× L4, PCIe; verify the GPU count on the EC2 instance-types page) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.12xlarge"` in `terraform.tfvars`) |
| Cost | **look up the on-demand $/h for `g6.12xlarge` in your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 2 instance-hours. Add gp3 storage for as long as the volume exists. Spot (`use_spot = true`) is fine here: every step can be rerun |
| Quota | "Running On-Demand G and VT instances" (or the Spot variant) ≥ the vCPUs of `g6.12xlarge` |
| Safety | no inbound ports; servers bind to `127.0.0.1` with an API key; reach them with `make forward` |

## Launch

```bash
cd infra/aws/single-node && cp -n terraform.tfvars.example terraform.tfvars   # set owner, budget_email, instance_type
make up && make ssm
```

## Run (on the instance)

```bash
git clone https://github.com/talalahmed311/silocon-to-serving && cd silocon-to-serving
export S2S_API_KEY=$(openssl rand -hex 16)
uv sync && uv pip install -r env/requirements-serving.txt
export MODEL=/opt/dlami/nvme/m          # your pinned 8B model, delivered with P3.7's cache (weights_bucket in tfvars)
bash course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/03_vllm_tp.sh          # ~45 min
torchrun --nproc-per-node 4 course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/04_fsdp2_toy.py | tee results/p4.2/fsdp2.txt
torchrun --nproc-per-node 4 course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/04_fsdp2_toy.py --ddp | tee results/p4.2/ddp.txt
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.
