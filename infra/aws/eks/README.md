# infra/aws/eks: the T3 cluster for P3 and P4

| | |
|---|---|
| Creates | a VPC (2 AZs, private subnets, a single NAT), an S3 gateway endpoint, EKS (`terraform-aws-eks` **v21.26.0**) with a `system` node group (1× m6i.large) and a tainted `gpu` node group (AL2023 NVIDIA AMI, **min 0**), and a budget |
| Add-ons | `addons.sh`: GPU Operator v26.7.1, kube-prometheus-stack, KEDA 2.21.0 (chart versions UNVERIFIED pins) |
| Access | the EKS API is public but restricted to `allowed_api_cidrs` (your /32). Workloads are reached via `kubectl port-forward` or an authenticated ingress (P3.8) |
| Cost | **look up in-region prices**: the EKS control plane $/h + NAT gateway $/h + per-GB processing + m6i.large + GPU nodes while they run. The control plane and the NAT cost money **even when the GPU pool is at zero**: `make down` when you stop for the day |
| Teardown | `make down`, which deletes LoadBalancers and Helm releases first, then `terraform destroy` |

```bash
cd infra/aws/eks && cp -n terraform.tfvars.example terraform.tfvars   # set owner, email, allowed_api_cidrs
make up && make addons
kubectl get nodes -L s2s/pool,nvidia.com/gpu.product
make down
```

**Module input names are UNVERIFIED** against v21.26.0. The build environment could only list the repository's files, not run Terraform. `make check` (or `tools/ci_terraform.sh`) runs `terraform validate` and tells you which names to fix.
