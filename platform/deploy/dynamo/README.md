# NVIDIA Dynamo on EKS (P3.4)

Dynamo (`ai-dynamo/dynamo@b37890b6`, **v1.5.0**) is NVIDIA's distributed inference framework. It does:

- **disaggregated serving**: separate prefill and decode workers that hand KV caches over the network (the DistServe idea)
- **KV-aware routing**: send a request to the worker that already holds its prefix
- a planner that rebalances prefill and decode capacity

It deploys to Kubernetes through its own operator and CRDs.

We don't duplicate Dynamo's manifests, because they move between releases. Follow the **Kubernetes deployment guide in the pinned repo**: look under `docs/` and `deploy/` at v1.5.0. Record exactly which files you used in your notes. The course's contract:

1. Install the operator and CRDs with the guide's Helm charts, at the pinned version.
2. Deploy the **aggregated** example with your model (one worker that does both phases). Measure it with #4: `loadgen.cli --rates …`.
3. Deploy the **disaggregated** example (1 prefill + 1 decode worker, which needs 2 GPUs: set `gpu_max: 2` in `infra/aws/eks`). Measure again at the same rates.
4. Fill in the comparison table in P3.4 exercise 4.

Every command and value in your run is UNVERIFIED until it has run, so paste the exact versions and file paths into `results/p34_dynamo.md`.
