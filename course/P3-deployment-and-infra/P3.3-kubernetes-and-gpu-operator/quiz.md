# P3.3 quiz

<details><summary><b>1.</b> Which Operator components are disabled on the EKS AL2023 NVIDIA AMI, and why?</summary>

The driver and the container toolkit. The AMI already provides both.
</details>

<details><summary><b>2.</b> Why does a GPU server need a <code>startupProbe</code> as well as a liveness probe?</summary>

Model load and graph capture take minutes. Without a startup probe, the liveness probe would kill the pod before it ever becomes healthy, forever.
</details>

<details><summary><b>3.</b> What does <code>preStop: sleep 20</code> buy during a rollout?</summary>

Time for endpoint removal to reach every proxy before the process starts shutting down, so no new requests hit a dying pod. In-flight streams then finish within the grace period.
</details>

<details><summary><b>4.</b> Why put a taint on GPU nodes <i>and</i> a toleration on GPU pods?</summary>

The taint keeps everything else off the expensive nodes, so they can scale to zero. The toleration lets GPU pods in. The nodeSelector or affinity makes them *prefer* or *require* those nodes.
</details>

<details><summary><b>5.</b> What does GPU Feature Discovery add?</summary>

Node labels describing the GPU (product, memory, compute capability, MIG and sharing state), so pods can select hardware by property.
</details>
