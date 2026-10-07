# P5.8 quiz

<details><summary><b>1.</b> Why is naive attention memory-bound at long N even though attention is "a GEMM"?</summary>

It writes and rereads the B·H·N² score/probability matrix several times. That traffic grows as N² while the inputs grow as N, and the softmax between the two GEMMs forces the round-trip through HBM.
</details>

<details><summary><b>2.</b> Write the update of (m, l, o) when a new tile with scores s_j arrives.</summary>

m' = max(m, max s_j); o' = o·e^{m−m'} + Σ e^{s_j−m'}·v_j; l' = l·e^{m−m'} + Σ e^{s_j−m'}; output o/l at the end.
</details>

<details><summary><b>3.</b> What does FA-2 change in the loop order, and why does it help?</summary>

Q tiles become the outer loop, one per thread block, so (m, l, o) stay in registers for the whole K/V sweep. That eliminates FA-1's HBM read-modify-write of O per tile, and adds parallelism across query tiles.
</details>

<details><summary><b>4.</b> How much work does causal tile skipping save?</summary>

About half: K/V tiles strictly after a Q tile's last row are fully masked and skipped.
</details>

<details><summary><b>5.</b> Why is decode attention memory-bound, and what does paging add?</summary>

One query per sequence means ~2 FLOPs per KV byte read. Paging adds an indirection (block table) per KV block, which is cheap if a whole block is read contiguously.
</details>
