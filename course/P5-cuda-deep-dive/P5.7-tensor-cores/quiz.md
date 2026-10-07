# P5.7 quiz

<details><summary><b>1.</b> How many FMAs does one m16n8k16 mma do, versus one warp-wide FFMA instruction?</summary>

16·8·16 = 2,048 multiply-adds per mma instruction (per warp) vs 32 for FFMA. Tensor-core peak comes from that ratio, not from higher clocks.
</details>

<details><summary><b>2.</b> In the m16n8 accumulator layout, which C elements does lane 13 hold?</summary>

g = 13/4 = 3, t = 13%4 = 1: rows 3 and 11, columns 2 and 3 of the 16×8 tile.
</details>

<details><summary><b>3.</b> Why pad shared-memory rows by 16 bytes for ldmatrix?</summary>

ldmatrix reads 8 rows of 16 bytes each. With a power-of-two row stride they'd fall into the same banks. A 16-byte pad spreads them over distinct banks.
</details>

<details><summary><b>4.</b> What does `cp.async.wait_group 1` guarantee?</summary>

At most one committed group is still pending, so all older groups (the tile about to be consumed) have completed for this thread. A `__syncthreads()` then makes them visible to the whole block.
</details>

<details><summary><b>5.</b> Why is the B tile loaded with `ldmatrix … .trans`?</summary>

The mma wants B as a K-major ("col") fragment, but the tile is stored row-major (N contiguous). The transposing load produces the column fragment directly from row-major shared memory.
</details>
