# P4.4 quiz

<details><summary><b>1.</b> Which sharing mode gives memory and fault isolation?</summary>

MIG only. Time-slicing and MPS share one memory space and one fault domain.
</details>

<details><summary><b>2.</b> Why can MIG strand capacity?</summary>

Tenants must fit fixed profiles. A 12 GB tenant needs a 20 GB profile on an A100-80GB, and placement rules can leave slices unusable.
</details>

<details><summary><b>3.</b> Per-tenant tok/s = 30, 30, 30, 10. What's Jain's index?</summary>

(100)² / (4 · (900·3 + 100)) = 10,000 / 11,200 ≈ 0.89.
</details>

<details><summary><b>4.</b> What happens to pods when you change a GPU's MIG layout?</summary>

The GPU has to be idle: the MIG manager drains and evicts the GPU pods, reconfigures, and the device plugin re-advertises the new resources.
</details>

<details><summary><b>5.</b> When is MPS the right choice?</summary>

Many small models or workers from one trust domain whose kernels alone don't fill the GPU. Concurrency raises utilization, and isolation is less of a concern.
</details>
