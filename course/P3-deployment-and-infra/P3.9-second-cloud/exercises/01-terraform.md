# Exercise 1: Terraform for a GKE GPU node pool (T0 → T3)

1. Read `infra/gcp/gke/main.tf` next to `infra/aws/eks/main.tf`. For each resource, name its AWS counterpart. Use the README's mapping table and fill in the gaps it leaves out (the router, the node service account and its three roles).
2. `bash tools/ci_terraform.sh` runs `fmt -check` + `validate` on it with no credentials. `TODO(run)`.
3. `test_gke_tf.py` is a static policy check: private nodes, an authorized-networks list (and never `0.0.0.0/0`), a min-0 GPU pool, `GKE_METADATA` on every pool, a billing budget, and `deletion_protection = false` so `make down` works. Make one of its checks fail on purpose to see the message, then revert.
4. **Your task:** add a second GPU pool with `spot = true` and a higher autoscaling max. Then add a `test_gke_tf.py` check that every GPU pool has `min_node_count = 0`. This is the GKE version of P3.6's spot + on-demand pair. How do you make the scheduler *prefer* spot without Karpenter-style weights? (Hint: node affinity `preferredDuringScheduling` on `cloud.google.com/gke-spot`, UNVERIFIED label name.)
5. `TODO(run-on: GCP)`: `make plan`, then `make up`. Record the time from `apply` to a Ready GPU node, alongside your EKS number from P3.2.
