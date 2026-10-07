# infra/aws/distributed: P4.3 add-on to the EKS cluster

It adds the pieces #9 needs on top of [`infra/aws/eks`](../eks/README.md):

| Resource | Why |
|---|---|
| S3 bucket (private, versioned, SSE, lifecycle expiry 14 d, `force_destroy` for labs) | DCP checkpoints and Ray run state survive pod and node loss |
| IAM role + **EKS Pod Identity** association for `s2s-train/s2s-train` | only the training ServiceAccount can touch the bucket. No keys anywhere |
| `kuberay.sh` | the KubeRay operator, plus the `s2s-train` namespace and ServiceAccount |

GPU capacity comes from the EKS cluster's GPU node group. For P4.3, set `gpu_instance_types = ["g6.12xlarge"]` (4× L4) and `gpu_max = 2` in `infra/aws/eks/terraform.tfvars`. The Pod Identity agent add-on must be enabled on the cluster (EKS docs, UNVERIFIED add-on name `eks-pod-identity-agent`).

| | |
|---|---|
| Cost | the EKS cluster (control plane, system nodes, NAT) + **GPU node-hours** at the `g6.12xlarge` price in your region (`____ $/h`) + S3 storage for checkpoints (small). Plan ≈ 2 GPU-node-hours for P4.3 |
| Teardown | `kubectl delete rayjob --all -n s2s-train`, then `make down` here, then `make down` in `infra/aws/eks`. Check with `../scripts/cost.sh` |
| Auto-stop | the RayJob has `activeDeadlineSeconds: 7200` and `shutdownAfterJobFinishes: true`. The GPU node group is min-0, so nodes go away when the job does |
| Safety | no public endpoints: the Ray dashboard is reached with `kubectl port-forward`. The bucket blocks all public access |

```bash
cp -n terraform.tfvars.example terraform.tfvars && make up && make kuberay
kubectl apply -f ../../../platform/training/ray/rayjob.yaml    # after setting the bucket and image in it
```

**Multi-node with EFA** (p4d/p5) needs EFA-enabled node groups, the EFA device plugin and the `aws-ofi-nccl` plugin in the image. That's out of scope here. Follow awsome-distributed-training `d4325212` (its EKS and ParallelCluster directories).
