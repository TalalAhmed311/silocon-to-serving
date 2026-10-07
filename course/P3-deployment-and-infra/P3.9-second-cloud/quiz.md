# P3.9 quiz

<details><summary><b>1.</b> What's the GKE equivalent of IRSA, and what's the equivalent of the IMDSv2 hop limit?</summary>

Workload Identity Federation for GKE (a KSA mapped to IAM). `GKE_METADATA` mode on node pools, which hides node credentials from pods.
</details>

<details><summary><b>2.</b> Why does the GKE path skip the GPU Operator?</summary>

GKE installs the NVIDIA driver (`gpu_driver_installation_config`) and its own device plugin. A second driver stack would conflict.
</details>

<details><summary><b>3.</b> Why compare $/1M tokens at the SLO rather than $/GPU-hour?</summary>

The same GPU can deliver different throughput (CPU, topology, driver), and utilization dominates the real cost. $/1M tokens at the SLO captures both.
</details>

<details><summary><b>4.</b> Spot on GCP gives a shorter notice than on AWS. What do you change?</summary>

A shorter `terminationGracePeriodSeconds` and preStop, a lower max response length on spot replicas, and a larger share of on-demand for streaming-heavy traffic.
</details>

<details><summary><b>5.</b> Where should cloud-specific differences live in the repo?</summary>

In `infra/<cloud>/` and one Kustomize overlay. Never in platform code or the base manifests.
</details>
