# P6.1 on AWS

Exercise 2 only. Everything else is T0.

| | |
|---|---|
| Instance | `g6.xlarge` (1× L4) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) |
| Cost | look up the on-demand $/h for `g6.xlarge` in your region: `____ $/h`. ≈ 1.5 instance-hours. Spot is fine |
| Safety | no inbound ports; work in the SSM session (`make ssm`); `make down` at the end |

```bash
cd infra/aws/single-node && make up && make ssm
# on the instance:
git clone https://github.com/GeeeekExplorer/nano-vllm && cd nano-vllm && git checkout bb823b3e   # pinned SHA (verify)
pip install -e .            # then add your timers (exercise 2) and run its example script
```

Teardown: `make down`, then confirm in the console that no instance or volume is left. TODO(run-on: g6.xlarge).
