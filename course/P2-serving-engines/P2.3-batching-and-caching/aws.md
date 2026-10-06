# P2.3 on AWS

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4 24 GB) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.xlarge"` in `terraform.tfvars`) |
| Cost | **look up the on-demand $/h for `g6.xlarge` in your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 4 instance-hours. Add gp3 storage for as long as the volume exists. Spot (`use_spot = true`) is fine here: every step can be rerun |
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
bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model>                 # prefix caching on
PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 --duration 60 \
  --shared-prefix 1000 --shared-fraction 0.8 --out results/prefix_on.json
pkill -f "vllm serve"; bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model> --no-enable-prefix-caching
PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 1 2 4 8 16 --duration 60 \
  --shared-prefix 1000 --shared-fraction 0.8 --out results/prefix_off.json
PYTHONPATH=platform python -m loadgen.plot results/prefix_on.json results/prefix_off.json
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.
