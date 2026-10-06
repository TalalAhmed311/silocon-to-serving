# Exercise 4 — Schedule only on GPUs with ≥ 24 GB (T3)

1. On EKS with the GPU Operator, dump a GPU node's labels: `kubectl get node <gpu-node> -o json | jq '.metadata.labels | with_entries(select(.key|startswith("nvidia.com")))'`. Record the real names and units of the product, memory and compute-capability labels in your notes and in `platform/deploy/README.md`.
2. Add a `nodeAffinity` to `vllm.yaml` (in an overlay) requiring ≥ 24 GB of GPU memory, using the label you found.
3. Add a second GPU node group with a smaller GPU, or change the instance type, and show that the pod refuses to land there (`kubectl describe pod` shows the affinity mismatch).
