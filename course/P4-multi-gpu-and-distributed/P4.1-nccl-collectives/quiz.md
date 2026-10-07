# P4.1 quiz

<details><summary><b>1.</b> How many steps does a ring all-reduce on n ranks take, and how many bytes does each rank send?</summary>

2(n−1) steps. Each rank sends 2(n−1)/n · S bytes: reduce-scatter (n−1)/n · S, then all-gather (n−1)/n · S.
</details>

<details><summary><b>2.</b> nccl-tests reports algbw 100 GB/s for all_reduce on 8 GPUs. What's busbw, and what do you compare it with?</summary>

busbw = 100 × 2·7/8 = 175 GB/s. Compare it with the per-GPU link bandwidth from a cited spec.
</details>

<details><summary><b>3.</b> When does a tree all-reduce beat a ring?</summary>

For small messages on many ranks: the ring's latency term 2(n−1)α dominates, and the tree's is about 2 log₂ n · α.
</details>

<details><summary><b>4.</b> Why is FSDP's communication volume the same as DDP's?</summary>

DDP all-reduces gradients. FSDP does a reduce-scatter of gradients plus an all-gather of parameters, and RS + AG moves the same bytes as one all-reduce.
</details>

<details><summary><b>5.</b> A decode-time TP all-reduce is 8 KB. Is it latency- or bandwidth-bound?</summary>

Latency-bound: S/B is nanoseconds at NVLink speeds, while α is microseconds. That's why TP helps decode less than the FLOP math suggests, and why NCCL's low-latency protocols and custom all-reduce kernels exist.
</details>
