# P3.7 on AWS

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.xlarge"` in `terraform.tfvars`) |
| Storage | the instance-store NVMe path above is an assumption (UNVERIFIED for your AMI: check `lsblk`); otherwise use the root gp3 volume and note its provisioned throughput. Delete the registry bucket objects when done: S3 storage bills monthly |
| Cost | **look up the on-demand $/h for `g6.xlarge` in your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 2 instance-hours. Add gp3 storage for as long as the volume exists. Spot (`use_spot = true`) is fine here: every step can be rerun |
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
uv sync --extra torch && uv pip install boto3   # S3Store; not in the T0 lockfile on purpose
# The registry: a private, versioned bucket you own, set as weights_bucket / weights_prefix="models/" in terraform.tfvars.
# The instance role is READ-ONLY on it (iam.tf). Upload once from your laptop with your own credentials:
#   huggingface-cli download <pinned 8B model> --revision <sha> --local-dir m
#   PYTHONPATH=platform python -m weights.manifest build m --model llama3-8b --version <sha>
#   aws s3 sync m s3://<bucket>/models/llama3-8b/<sha>/
export REG=s3://REPLACE-your-registry-bucket/models
PYTHONPATH=platform uv run python -c "from weights.cache import WeightCache, S3Store; print(WeightCache('/opt/dlami/nvme/cache', 200<<30, S3Store('$REG')).get('llama3-8b', '<sha>'))"
ln -s /opt/dlami/nvme/cache/llama3-8b/<sha> /opt/dlami/nvme/m
sudo -E PYTHONPATH=platform uv run python -m weights.bench --dir /opt/dlami/nvme/m --s3 $REG/llama3-8b/<sha> --gpu
# streamer vs default (exercise 4): time to "Application startup complete" in each log
uv pip install -r env/requirements-serving.txt
( time vllm serve /opt/dlami/nvme/m --host 127.0.0.1 --api-key $S2S_API_KEY ) 2>&1 | tee results/p3.7-default.log
( time vllm serve $REG/llama3-8b/<sha> --load-format runai_streamer --host 127.0.0.1 --api-key $S2S_API_KEY ) 2>&1 | tee results/p3.7-streamer.log
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.
