# 03_ray: #9 as a RayJob on EKS (primary path, T3)

The job spec and launcher live with the project, at [`platform/training/ray/`](../../../../../platform/training/ray/rayjob.yaml). The infra is in [`infra/aws/distributed`](../../../../../infra/aws/distributed/README.md). Steps:

```bash
cd infra/aws/eks && make up && make addons                     # with gpu_instance_types = ["g6.12xlarge"]
cd ../distributed && make up && make kuberay
# image: FROM rayproject/ray-ml:2.59.0-gpu (pin by digest) + COPY this repo to /app; push to ECR; set it in rayjob.yaml
kubectl apply -f platform/training/ray/rayjob.yaml
kubectl -n s2s-train get rayjob s2s-train -w
kubectl -n s2s-train port-forward svc/s2s-train-head-svc 8265:8265   # Ray dashboard on localhost only (service name UNVERIFIED)
```

Fault injection: `bash ../04_fault_inject.sh ray` deletes a worker pod mid-run. Ray restarts the worker group, and `train.py` resumes from the latest committed checkpoint.
