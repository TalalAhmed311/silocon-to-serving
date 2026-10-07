# P3.6 on AWS (T3, the EKS cluster from P3.2)

| | |
|---|---|
| Cluster | [`infra/aws/eks`](../../../infra/aws/eks/README.md) with its tainted, min-0 GPU node group (`g6.xlarge`, 1× L4) |
| Cost | control plane per-hour fee + system nodes + **GPU node-hours while scaled up**. Look up all three in your region: `____ $/h`. This module needs ≈ 3 GPU-node-hours. NAT gateway hours count too |
| Quota | "Running On-Demand G and VT instances" ≥ 4 vCPU × max GPU nodes. For the spot exercise, also the Spot variant |
| Safety | the vLLM Service stays `ClusterIP`. Reach it with `kubectl port-forward`. The HF token comes from Secrets Manager via the P3.2 addon, never from a manifest |

## Steps

```bash
cd infra/aws/eks && make up && make addons      # P3.2: cluster, GPU Operator, kube-prometheus-stack, KEDA
kubectl apply -k platform/deploy/overlays/eks
kubectl apply -f platform/autoscaler/scaledobject.yaml
# terminal 2: record the timeline
kubectl -n s2s get pods,hpa -w --output-watch-events -o wide | ts '%.s' | tee results/p3.6-watch.log
# terminal 3: ramp with #4 through the port-forwarded gateway
uv run python -m loadgen.cli --url http://localhost:9000 --rates 1 4 8 16 --duration 300 --out results/p3.6-ramp.json
uv run python -m autoscaler.coldstart --namespace s2s --pod <the pod that scaled up> | tee results/p3.6-coldstart.md
```

Optional, for exercise 4 on real spot: install Karpenter with an IRSA/Pod Identity role and the `EC2NodeClass` named `gpu` from its docs. Apply `platform/autoscaler/karpenter-nodepools.yaml`. Then trigger an interruption with AWS FIS (`aws:ec2:send-spot-instance-interruptions`, UNVERIFIED action name). Run the ramp at the same time and count failed requests.

## Teardown

```bash
kubectl delete -f platform/autoscaler/scaledobject.yaml
cd infra/aws/eks && make down && ../scripts/cost.sh
```

**Auto-stop:** the GPU node group is min-0, so with no pods it scales back to zero. The control plane and NAT keep billing until `make down`, and the budget alert from `infra/aws/guardrails` is your backstop.
