# P6.7 on AWS (also used by P6.3 ex. 4, P6.5 ex. 1/3, P6.6)

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4, 24 GB) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) |
| Cost | look up the on-demand $/h for `g6.xlarge` in your region: `____ $/h`. P6.7 needs ≈ 6 instance-hours (parity, profiling, two load tests), P6.3/P6.5/P6.6 ≈ 1–2 h each. Spot is fine: every step can be rerun. `make stop` between sessions keeps the disk and stops the instance charge |
| Quota | "Running On-Demand G and VT instances" ≥ 4 vCPUs |
| Safety | no inbound ports; servers bind to `127.0.0.1` with `S2S_API_KEY`; reach them with `make forward`. Model weights come from Hugging Face with your token in SSM Parameter Store, never in the repo |

```bash
cd infra/aws/single-node && make up && make ssm
# on the instance
git clone https://github.com/talalahmed311/silocon-to-serving && cd silocon-to-serving
uv sync --extra torch && uv pip install "vllm==0.31.0"          # pinned; check the CUDA/torch pairing in vLLM's docs
uv run --extra torch pytest platform/engine/v1/tests -q          # includes the gpu tests on this box
MODEL_DIR=build/my-model HF_MODEL=<model id> bash course/P6-engine-internals/P6.7-project-tiny-engine-gpu/bench/compare.sh
```

Copy `results/p6.7/` back before teardown (`tar czf - results | base64` in the SSM session). Then `make down` and check the console for leftover instances and volumes. TODO(run-on: g6.xlarge).
