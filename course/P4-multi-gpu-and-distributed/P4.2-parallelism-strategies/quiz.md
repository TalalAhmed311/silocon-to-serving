# P4.2 quiz

<details><summary><b>1.</b> Why column-split the first MLP matmul and row-split the second?</summary>

The nonlinearity is elementwise over intermediate units, so after a column split each rank can apply it locally. The row split then gives partial outputs that a single all-reduce sums.
</details>

<details><summary><b>2.</b> How many all-reduces does Megatron TP do per transformer layer in the forward pass?</summary>

Two: one after attention (W_o) and one after the MLP (W_down).
</details>

<details><summary><b>3.</b> p = 8 stages, m = 8 microbatches. What's the bubble? What m gets it under 10%?</summary>

7/15 ≈ 47%. You need (7)/(m+7) < 0.1, so m > 63.
</details>

<details><summary><b>4.</b> What does 1F1B improve over GPipe, and what does it not?</summary>

It bounds in-flight activations to about p microbatches instead of m, which saves memory. The bubble stays the same (interleaving is what reduces it).
</details>

<details><summary><b>5.</b> Bytes per parameter for FSDP on 8 GPUs with mixed-precision Adam?</summary>

16/8 = 2 B per param per GPU (params, grads and optimizer state all sharded), plus activations and temporary all-gathered layers.
</details>

<details><summary><b>6.</b> An 8B model fits on one L4. You have 4 L4s on PCIe. TP = 4 or 4 replicas, for max throughput?</summary>

4 replicas. There's no communication, each replica gets the full KV budget of its GPU, and throughput scales linearly. TP = 4 helps per-token latency at low load, if at all on PCIe.
</details>
