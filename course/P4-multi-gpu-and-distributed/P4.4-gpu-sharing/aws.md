# P4.4 on AWS (T3): the #11 contention matrix

| | |
|---|---|
| Instance | `p4d.24xlarge` (8× A100 40 GB, MIG-capable) via [`infra/aws/single-node`](../../../infra/aws/single-node/README.md) (`instance_type = "p4d.24xlarge"`). Only **one** GPU is used, but there's no smaller AWS instance with a MIG-capable GPU (UNVERIFIED: check the current EC2 GPU instance list, as newer types may exist). Time-slicing and MPS alone also work on `g6.xlarge` (no MIG) |
| Cost | **expensive**: look up the `p4d.24xlarge` on-demand price, and prefer an **EC2 Capacity Block for ML** (a short fixed reservation). Budget ≈ 1.5 h. Write it here: `____ $/h` |
| Quota | "Running On-Demand P instances" ≥ 96 vCPUs (usually needs a request, and approval can take days), or a Capacity Block purchase |
| Safety | no inbound ports. Servers bind to `127.0.0.1` with an API key. Reach them with `make forward` |

## Run (on the instance)

```bash
git clone https://github.com/talalahmed311/silocon-to-serving && cd silocon-to-serving
export S2S_API_KEY=$(openssl rand -hex 16) MODEL=<a ~1B model dir, pinned by revision>
uv sync && uv pip install -r env/requirements-serving.txt
for m in timeslice mps; do for n in 2 4; do MODE=$m N=$n bash platform/partitioning/contention.sh; done; done
sudo nvidia-smi -i 0 -mig 1 && sudo nvidia-smi mig -i 0 -cgi 3g.20gb,3g.20gb -C     # A100-40GB profile names (UNVERIFIED)
MODE=mig N=2 bash platform/partitioning/contention.sh
sudo nvidia-smi mig -i 0 -dci && sudo nvidia-smi mig -i 0 -dgi && sudo nvidia-smi mig -i 0 -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb -C
MODE=mig N=4 bash platform/partitioning/contention.sh
sudo nvidia-smi mig -i 0 -dci && sudo nvidia-smi mig -i 0 -dgi && sudo nvidia-smi -i 0 -mig 0
```

A `1g.5gb` slice only fits a small model. Size `MODEL` and `--max-model-len` to it. Copy `results/p4.4/` back before teardown.

## Teardown

```bash
make down && ../scripts/cost.sh       # in infra/aws/single-node
```

**Auto-stop:** the idle watchdog stops the instance after 30 idle minutes, and the CloudWatch backstop after 3 h. With a Capacity Block, the reservation ends on its own, but `make down` still deletes the volume.
