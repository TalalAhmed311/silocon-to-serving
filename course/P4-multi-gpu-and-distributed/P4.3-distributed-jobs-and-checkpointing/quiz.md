# P4.3 quiz

<details><summary><b>1.</b> Why write a COMMITTED marker after a barrier instead of trusting the checkpoint directory?</summary>

A crash mid-save leaves a directory with missing shards. The marker, written by rank 0 only after every rank has finished, makes "visible to resume" equal to "complete".
</details>

<details><summary><b>2.</b> How can a checkpoint saved on 2 ranks load on 4?</summary>

DCP records each tensor's global shape and its shard locations in `.metadata`. On load, each new rank reads the byte ranges its new shard needs.
</details>

<details><summary><b>3.</b> C = 30 s, M = 4 h. What's Young's τ*?</summary>

√(2·30·14,400) ≈ 930 s ≈ 15.5 min.
</details>

<details><summary><b>4.</b> What's the expected RPO with interval τ?</summary>

About τ/2 of compute, plus anything lost during an interrupted save.
</details>

<details><summary><b>5.</b> Why must the data stream be keyed by global sequence index, not by rank?</summary>

So a resumed run at a different world size, or after a restart, sees exactly the batches it would have seen. Otherwise resharding silently changes the data order and reproducibility is lost.
</details>
