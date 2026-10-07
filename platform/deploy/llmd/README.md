# llm-d on EKS (P3.4)

llm-d (`llm-d/llm-d@23f57388`, **v0.10.0**) is a Kubernetes-native distributed inference stack built around vLLM. It provides:

- an **inference scheduler**: an Envoy-based endpoint picker that routes on KV-cache and prefix locality and on load
- optional prefill/decode disaggregation
- "well-lit path" Helm charts

As with Dynamo, follow the **pinned repo's quickstart and guides** (`docs/` and `guides/` at v0.10.0, or wherever they live at that tag) instead of copies here. The course's contract:

1. Deploy the basic inference-scheduling guide with your model on 2 GPU replicas.
2. Run #4 with a **shared-prefix** workload (`--shared-prefix 1000 --shared-fraction 0.8`) through llm-d's gateway, and again through a plain Kubernetes Service (round-robin) over the same 2 replicas.
3. Compare prefix-cache hit rates (vLLM's `/metrics` on each replica) and TTFT. Prefix-aware routing should keep a conversation on the replica that holds its blocks.
