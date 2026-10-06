# Exercise 2: same manifests, only an overlay changes (T0 → T3)

`test_overlays.py` compares `platform/deploy/overlays/eks` and `overlays/gke`:

- the same `resources` (the same base manifests)
- the same gateway config (except the comment line)
- no `patches` in either overlay, with the only difference being `images`

1. Make it pass (it should as shipped). Then add a GKE-only patch that changes the vLLM `nodeSelector`, and watch it fail. **Why is that the right outcome?** The GPU pool label `s2s/pool: gpu` is set in *both* Terraform roots precisely so the manifests don't need to know the cloud.
2. Some differences are real, for example a StorageClass for the weights PVC (gp3 vs pd-balanced). Put them in a `cloud.yaml` resource in each overlay, and extend the test to allow exactly that file to differ.
3. `TODO(run-on: GKE)`: `kubectl apply -k platform/deploy/overlays/gke`, then run the same #4 ramp you ran on EKS. Keep the outputs for exercise 3.
