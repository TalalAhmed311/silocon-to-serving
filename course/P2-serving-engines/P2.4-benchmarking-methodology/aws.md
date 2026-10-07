# P2.4 on AWS

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4 24 GB) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "g6.xlarge"` in `terraform.tfvars`) |
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
bash infra/aws/scripts/bootstrap.sh p2 <model-repo> <revision-sha> && source .venv-serving/bin/activate
bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<model>
# 1) your #4
PYTHONPATH=platform python -m loadgen.cli --url http://127.0.0.1:8000 --rates 2 4 8 --duration 60 --out results/p24_loadgen.json
# 2) GuideLLM at the same rates (check `guidellm benchmark --help` at v0.8.0 for the exact flags)
OPENAI_API_KEY=$S2S_API_KEY guidellm benchmark --target http://127.0.0.1:8000 --rate-type constant --rate 2,4,8 \
  --max-seconds 60 --data "prompt_tokens=512,output_tokens=128" --output-path results/p24_guidellm.json
# 3) vllm bench serve (random dataset, same shape)
OPENAI_API_KEY=$S2S_API_KEY vllm bench serve --base-url http://127.0.0.1:8000 --model /opt/models/<model> \
  --dataset-name random --random-input-len 512 --random-output-len 128 --request-rate 4 --num-prompts 240 \
  --save-result --result-dir results/
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.

> **Flags are UNVERIFIED** at the pinned versions: the build environment couldn't run them. If a flag is rejected, check `guidellm benchmark --help` and `vllm bench serve --help`, fix the command here, and note the change in your report.
