# P2.6 on AWS

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
bash infra/aws/scripts/bootstrap.sh p2 <target-repo> <sha> && source .venv-serving/bin/activate
.venv-serving/bin/huggingface-cli download <draft-repo-same-tokenizer> --revision <sha> --local-dir /opt/models/draft
# #10 prototype (no KV cache: compare methods within the harness)
PYTHONPATH=platform python -m specdec.bench --target /opt/models/<target> --draft /opt/models/draft --k 1 2 4 \
  --prompts course/P2-serving-engines/P2.6-speculative-decoding-and-cuda-graphs/examples/prompts_code.txt --tokens 64
# vLLM n-gram spec decode (flag format UNVERIFIED at v0.31.0 — check `vllm serve --help`)
bash course/P2-serving-engines/P2.1-first-serve-with-vllm/examples/01_serve.sh /opt/models/<target> \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'
python course/P2-serving-engines/P2.1-first-serve-with-vllm/bench/decode_ceiling.py --url http://127.0.0.1:8000 --gpu L4 \
  --model-config /opt/models/<target>/config.json
```

Copy `results/` back before teardown, for example `tar czf - results | base64` in the SSM session, then decode it locally.

## Teardown

```bash
make down            # in infra/aws/single-node, on your laptop
../scripts/cost.sh   # confirm nothing tagged Project=silicon-to-serving is still running
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes. The CloudWatch backstop stops it after 3 h of near-zero CPU. Only `make down` deletes the disk.
