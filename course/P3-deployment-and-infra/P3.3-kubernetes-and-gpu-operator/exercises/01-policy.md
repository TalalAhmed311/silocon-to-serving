# Exercise 1 — Manifests pass policy, and kubeconform (T0)

1. `test_manifests.py` runs `platform/deploy/policy.py` over every Deployment in `platform/deploy/`. It must stay green as you add the P3.4 stacks.
2. Install `kubeconform`, then run `kustomize build platform/deploy/overlays/kind | kubeconform -strict -summary`, and the same for `overlays/eks`. Add CRD schemas for the KServe, KEDA and Ray objects when they arrive (`-schema-location` with a CRD catalog).
3. Add one rule to `policy.py`, with a test: every container must set a CPU **request**. Without one, the scheduler can overpack a node and starve the tokenizer threads.
