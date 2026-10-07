# autoscaler: #3, the queue-based GPU autoscaler

| File | What |
|---|---|
| `scaledobject.yaml` | KEDA scales the vLLM Deployment on `running + waiting` requests per replica. A gateway in-flight trigger handles scale-from-zero |
| `karpenter-nodepools.yaml` | GPU NodePools: spot first (weight 100), on-demand fallback (weight 10), both tainted and labelled `s2s/pool=gpu` |
| `policy.py` | the scaling decision as a pure, tested function: immediate bounded scale-up, stabilized scale-down, cooldown before zero |
| `coldstart.py` | phase timings of a scale-from-zero, from Kubernetes events |

Built in [P3.6](../../course/P3-deployment-and-infra/P3.6-autoscaling/README.md).
