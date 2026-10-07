# P3.6 quiz

<details><summary><b>1.</b> Why is GPU utilization a poor autoscaling signal for LLM serving?</summary>

It reads near 100% as soon as one request is decoding, so it can't tell 1 user from 400. Running + waiting requests per replica tracks the actual load.
</details>

<details><summary><b>2.</b> Where does the scale-from-zero signal come from, and why?</summary>

From something that stays up at zero replicas, such as the gateway's in-flight/queue gauge. The engine's own metrics disappear with the last pod.
</details>

<details><summary><b>3.</b> Why scale up fast and down slowly?</summary>

Under-provisioning breaks the SLO right away, and new GPU capacity takes minutes (a cold start). Over-provisioning for a few minutes only costs money. Slow scale-down also avoids flapping.
</details>

<details><summary><b>4.</b> Name the four cold-start phases and one mitigation for each.</summary>

Node provisioning (warm pool, balloon pods, min_size ≥ 1) · image pull (pre-pull, bake into AMI, smaller image) · weight load (NVMe cache, streaming loader) · warm-up (fewer CUDA-graph sizes, cache compiled artifacts).
</details>

<details><summary><b>5.</b> A spot node gets its two-minute notice. Which requests can the gateway save, and which can't it?</summary>

It can save requests that haven't received a byte yet: they're retried elsewhere. Streams already in progress can't be retried transparently, so drain them within the grace period, keep them short on spot, or count them against the error budget.
</details>

<details><summary><b>6.</b> Why diversify instance types in the spot NodePool?</summary>

Interruptions and capacity shortages are per instance pool. Several types (and AZs) make it less likely that one event takes out every replica at once.
</details>
