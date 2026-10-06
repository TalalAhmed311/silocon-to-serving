# deploy: #1, the self-hosted inference cluster

These are the Kubernetes manifests for the platform. The same `base/` deploys to two places:

- **kind**, on a laptop, with `mockllm` as the backend (T0)
- **EKS**, with vLLM on GPUs (T3)

| Path | What |
|---|---|
| `kind/kind-config.yaml` | 3-node kind cluster with a labelled "gpu" pool |
| `gpu-operator-values.yaml` | GPU Operator v26.7.1 values for AL2023 NVIDIA nodes (driver and toolkit come with the AMI) |
| `base/vllm.yaml` | vLLM Deployment, Service and PDB: GPU request, taint toleration, startup/readiness/liveness probes, preStop drain, non-root read-only pod, API key from a Secret, `/dev/shm` sized |
| `base/mockllm.yaml` | the same shape for kind |
| `overlays/{kind,eks}` | kustomize overlays |
| `policy.py` | static policy checks run in CI and by P3.3's tests |
| `kserve/`, `rayserve/`, `dynamo/`, `llmd/` | the four serving stacks compared in P3.4 |

```bash
kind create cluster --name s2s --config platform/deploy/kind/kind-config.yaml
docker build -f env/Dockerfile.platform -t s2s-platform:dev . && kind load docker-image s2s-platform:dev --name s2s
kubectl apply -k platform/deploy/overlays/kind && kubectl -n s2s get pods -o wide
uv run python platform/deploy/policy.py
```
