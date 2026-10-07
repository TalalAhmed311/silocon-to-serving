# P5.6 quiz

<details><summary><b>1.</b> Rung 3 uses 32×32 tiles. By how much does it cut global loads versus rung 2?</summary>

Each loaded element of A and B is reused 32 times from shared memory, so about 32×. DRAM traffic per FLOP drops by the tile size.
</details>

<details><summary><b>2.</b> In rung 5 (8×8 per thread), how many FMAs does one k-step do per smem load?</summary>

8 + 8 = 16 loads feed 64 FMAs: 4 FMAs per load (vs about 1 in rung 3).
</details>

<details><summary><b>3.</b> Why store the A tile transposed in rung 6?</summary>

So the 8 values a thread needs for one k (a column of the A tile) are contiguous in shared memory and can be read with float4 loads.
</details>

<details><summary><b>4.</b> What does double buffering hide, and what does it cost?</summary>

It hides global-load latency behind compute of the previous tile, at the cost of twice the shared memory (which can lower occupancy) and more registers for the in-flight loads.
</details>

<details><summary><b>5.</b> How do you call column-major cuBLAS for row-major C = A·B?</summary>

Compute Cᵀ = Bᵀ·Aᵀ: `cublasSgemm(h, N, N, n, m, k, &1, B, n, A, k, &0, C, n)`.
</details>
