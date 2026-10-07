# P4.1 on AWS

| | |
|---|---|
| Instance | `g6.12xlarge` (4× L4, PCIe; verify the GPU count on the EC2 instance-types page) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.12xlarge"` in `terraform.tfvars`) |
| Cost | **look up the on-demand $/h for `g6.12xlarge` in your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 1.5 instance-hours. Add gp3 storage for as long as the volume exists. Spot (`use_spot = true`) is fine here: every step can be rerun |
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
bash course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/examples/03_nccl_tests.sh      # ~20 min
for c in all_reduce all_gather reduce_scatter; do
  python3 course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/bench/parse_nccl_tests.py results/nccl/$c.txt \
    --link-gbs <cited PCIe bandwidth> --json results/nccl/$c.json
done
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.

## Optional: the NVLink box (p4d.24xlarge)

The same script on `p4d.24xlarge` (8× A100 with NVSwitch) shows what NVLink changes. It's expensive: **look up its on-demand price and prefer an EC2 Capacity Block for ML** (a fixed short reservation). Budget ≈ 0.5 h. Launch it with `instance_type = "p4d.24xlarge"`, run only `03_nccl_tests.sh`, copy `results/nccl/` back, and run `make down` straight away. Quota: "Running On-Demand P instances" ≥ 96 vCPUs, which usually needs a quota request first.

Multi-node (EFA) runs are covered in P4.3's Ray-on-EKS guide, and nccl-tests over EFA needs the `aws-ofi-nccl` plugin (awsome-distributed-training `d4325212`).
