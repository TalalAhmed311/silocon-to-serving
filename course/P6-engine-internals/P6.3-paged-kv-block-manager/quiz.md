# P6.3 quiz

<details><summary><b>1.</b> Block size 16, a sequence of 100 tokens. Which slot holds position 37 if its block table is [7, 2, 9, 4, 0, 5, 1]?</summary>

37 // 16 = 2 → physical block 9; 37 % 16 = 5 → slot 9·16 + 5 = 149.
</details>

<details><summary><b>2.</b> What is the worst-case wasted KV memory per sequence?</summary>

block_size − 1 token slots (the unused tail of its last block). No external fragmentation, since every block is the same size.
</details>

<details><summary><b>3.</b> Why are only full blocks hashed?</summary>

A partial block is still being written; its content (and therefore its hash) would change. Matching it would hand another request K/V that later diverges.
</details>

<details><summary><b>4.</b> A finished request's hashed blocks have ref 0. Why not put them straight on the free list?</summary>

Then the next request with the same prefix couldn't reuse them. They stay in an LRU "cached-free" set: free for allocation if needed, matchable until then.
</details>

<details><summary><b>5.</b> When does copy-on-write trigger, and who performs the copy?</summary>

When a sequence appends into a partial last block with ref count > 1 (after a fork). The block manager allocates the destination and queues (src, dst); the model runner copies the K/V on the GPU before the next forward pass.
</details>
