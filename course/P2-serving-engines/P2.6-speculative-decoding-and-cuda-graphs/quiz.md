# P2.6 quiz

<details><summary><b>1.</b> Why does verifying k tokens cost about the same as generating one, at batch 1?</summary>

Batch-1 decode is memory-bound: the cost is reading the weights. A (k+1)-token pass reads them once and uses the spare compute for the extra positions.
</details>

<details><summary><b>2.</b> On rejection, why sample from norm(max(0, p − q)) and not from p?</summary>

Accepted proposals already contributed probability mass min(p, q). Sampling the remainder from the residual is exactly what makes the total distribution equal p. Sampling from p would over-weight tokens the draft also likes.
</details>

<details><summary><b>3.</b> α = 0.8, k = 4: expected tokens per target pass?</summary>

(1 − 0.8⁵)/(1 − 0.8) = (1 − 0.328)/0.2 ≈ **3.36**.
</details>

<details><summary><b>4.</b> Why can spec decoding reduce throughput at high batch?</summary>

At high batch the GPU's compute is already in use. Rejected proposals are wasted FLOPs that displace other sequences' useful work.
</details>

<details><summary><b>5.</b> What does a CUDA graph remove, and what does it require?</summary>

It removes per-kernel CPU launch overhead, by replaying a recorded sequence with one launch. It requires static shapes and buffer addresses, which is why vLLM pads to captured batch sizes, and no host sync inside the graph.
</details>
