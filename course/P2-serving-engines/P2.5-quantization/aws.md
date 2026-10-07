# P2.5 on AWS

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4 24 GB, FP8-capable) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.xlarge"` in `terraform.tfvars`) |
| Cost | **look up the on-demand $/h for `g6.xlarge` in your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 6 instance-hours. Add gp3 storage for as long as the volume exists. Spot (`use_spot = true`) is fine here: every step can be rerun |
| Quota | "Running On-Demand G and VT instances" (or the Spot variant) ≥ the vCPUs of `g6.xlarge` |
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
bash infra/aws/scripts/bootstrap.sh p2 <model-repo> <revision-sha> && source .venv-serving/bin/activate
uv pip install -p .venv-serving -r env/requirements-quant.txt
ln -sfn /opt/models/<model> /opt/models/base
python platform/bakeoff/quantize.py --base /opt/models/base --scheme fp8-dynamic --out /opt/models/base-fp8
python platform/bakeoff/quantize.py --base /opt/models/base --scheme awq-w4a16  --out /opt/models/base-awq
python platform/bakeoff/quantize.py --base /opt/models/base --scheme gptq-w4a16 --out /opt/models/base-gptq
python platform/bakeoff/run.py platform/bakeoff/variants.yaml
python platform/bakeoff/table.py results/bakeoff
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.
