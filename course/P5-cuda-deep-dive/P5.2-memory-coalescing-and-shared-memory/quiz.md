# P5.2 quiz

<details><summary><b>1.</b> A warp loads 32 floats at stride 4. How many sectors, and what efficiency?</summary>

The addresses span 32·4·4 = 512 bytes. Each sector holds 8 floats, so there are 2 useful floats per sector and 16 sectors. 8 of 32 bytes are useful: 25%.
</details>

<details><summary><b>2.</b> Why does `[32][33]` remove the column-read conflict?</summary>

Row i starts at word 33i, so element (i, k) is in bank (33i + k) mod 32 = (i + k) mod 32, which is distinct across lanes i.
</details>

<details><summary><b>3.</b> All 32 lanes read the same shared-memory word. Conflict?</summary>

No. It's a broadcast, served in one pass.
</details>

<details><summary><b>4.</b> Why does the smem transpose need `__syncthreads()` between the load and the store?</summary>

Each thread writes out elements that *other* threads loaded. Without the barrier, a thread may read a tile slot before its owner has written it.
</details>

<details><summary><b>5.</b> When does `float4` not help?</summary>

When the kernel is already at the bandwidth limit with scalar loads and has no instruction bottleneck, or when the alignment can't be guaranteed.
</details>
