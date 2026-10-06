# P2.1 on AWS

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4 24 GB) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md). Budget path: `g4dn.xlarge` (T4 16 GB) with a ≤ 3B model |
| Cost | **look up the on-demand $/h for your region** on the EC2 pricing page and write it here: `____ $/h`. This module needs ≈ 3–4 instance-hours. Add gp3 storage for as long as the volume exists |
| Quota | "Running On-Demand G and VT instances" ≥ 4 vCPU (see the single-node README) |
| Safety | no inbound ports; vLLM binds to `127.0.0.1` with `--api-key`; reach it with `make forward` |

## Launch

```bash
cd infra/aws/single-node && cp -n terraform.tfvars.example terraform.tfvars   # edit owner, email, instance_type
make up && make ssm
```

## Run (on the instance)

```bash
git clone https://github.com/talalahmed311/silocon-to-serving && cd silocon-to-serving
bash infra/aws/scripts/bootstrap.sh p2 <model-repo> <revision-sha>
source .venv-serving/bin/activate
export S2S_API_KEY=$(openssl rand -hex 16)          # this session only
bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model>
python course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py --url http://127.0.0.1:8000 \
  --api-key "$S2S_API_KEY" --gpu L4 --model-config /opt/models/<model>/config.json
vllm bench latency --model /opt/models/<model> --batch-size 1 --input-len 128 --output-len 128   # exercise 4 reference
```

Copy `results/` back before you tear down. For example, `tar czf - results | base64` in the SSM session, then decode it locally. Or `aws s3 cp` to your own bucket, after adding write permission yourself.

## Teardown

```bash
make down          # in infra/aws/single-node, on your laptop
../scripts/cost.sh # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes (GPU < 5%, no SSM session). The CloudWatch backstop stops it after 3 h of near-zero CPU. Neither one *deletes* the disk: only `make down` does.
