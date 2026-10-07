# P4.3 on AWS (T3): #9 as a RayJob on EKS

| | |
|---|---|
| Cluster | [`infra/aws/eks`](../../../infra/aws/eks/README.md) with `gpu_instance_types = ["g6.12xlarge"]` (4× L4 per node; verify the GPU count on the EC2 page) and `gpu_max = 2`, plus [`infra/aws/distributed`](../../../infra/aws/distributed/README.md) (checkpoint bucket, Pod Identity, KubeRay) |
| Cost | EKS control plane + system nodes + NAT + **g6.12xlarge node-hours**. Look up all of them in your region: `____ $/h`. Plan ≈ 2 GPU-node-hours (3 failure drills of ~20 min plus setup) |
| Quota | "Running On-Demand G and VT instances" ≥ 48 vCPU per `g6.12xlarge` node |
| Safety | no public endpoints (Ray dashboard via `kubectl port-forward` only). S3 access through the `s2s-train` ServiceAccount's Pod Identity role only. Images pinned by digest |

## Steps

```bash
cd infra/aws/eks && make up && make addons
cd ../distributed && cp -n terraform.tfvars.example terraform.tfvars && make up && make kuberay
# build + push the training image (FROM rayproject/ray-ml:2.59.0-gpu, COPY the repo to /app), set it and the bucket in:
kubectl apply -f platform/training/ray/rayjob.yaml
kubectl -n s2s-train logs -f -l ray.io/node-type=head | tee results/p4.3/ray-run.log     # label UNVERIFIED
bash course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/04_fault_inject.sh ray
```

Node-loss drill: in the EC2 console, terminate the GPU instance. Record the time. Watch `kubectl get nodes -w` and the head logs, and record when training logs its next step.

## Teardown

```bash
kubectl -n s2s-train delete rayjob --all
cd infra/aws/distributed && make down      # deletes the checkpoint bucket (force_destroy) — copy results first
cd ../eks && make down && ../scripts/cost.sh
```

**Auto-stop:** `activeDeadlineSeconds: 7200` on the RayJob, `shutdownAfterJobFinishes: true`, and a min-0 GPU node group. The control plane and NAT bill until `make down`, with the budget alert as the backstop.
