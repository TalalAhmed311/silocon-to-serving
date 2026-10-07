# P3.4 quiz

<details><summary><b>1.</b> What does disaggregated serving remove, and what does it add?</summary>

It removes interference between compute-bound prefill and memory-bound decode on the same GPU. It adds KV-cache transfer (bandwidth and latency) and the operational complexity of two pools to balance.
</details>

<details><summary><b>2.</b> Why does round-robin hurt prefix-cache hit rates?</summary>

It scatters requests that share a prefix (for example the turns of one conversation) across replicas, so each cache holds fragments and few requests hit.
</details>

<details><summary><b>3.</b> When would you pick Ray Serve over a plain Deployment?</summary>

When you need Python-level composition (multi-model pipelines, custom pre/post-processing) and autoscaling on ongoing requests inside one framework, and you're willing to run a Ray cluster.
</details>

<details><summary><b>4.</b> Why rendezvous hashing for session affinity rather than <code>hash(session) % N</code>?</summary>

With modulo, a change in N remaps almost every session. Rendezvous hashing moves only the sessions of the backend that left.
</details>
