# P6.4 quiz

<details><summary><b>1.</b> The tree holds "SYSTEMhello". You match "SYSTEMhi". What changes in the tree?</summary>

The edge "SYSTEMhello" is split into "SYSTEMh" → "ello"; the match returns 7 tokens and the "SYSTEMh" node. Nothing is inserted until the request finishes.
</details>

<details><summary><b>2.</b> Why can only leaves be evicted?</summary>

An internal node's tokens are a prefix of its children's sequences; their K/V was computed attending to it. Removing it would orphan the children's KV.
</details>

<details><summary><b>3.</b> Why does `lock` walk all the way to the root?</summary>

Evicting any ancestor would make the locked node unreachable (and, conceptually, invalidate its KV). Ref counts on the whole path keep every ancestor un-evictable while it's in use.
</details>

<details><summary><b>4.</b> Which matches more tokens: block hashing (block size 16) or a radix tree? When does it matter?</summary>

The radix tree, by up to 15 tokens per request (the partial last block). It matters for short prompts with short shared prefixes; for long shared prefixes the difference is negligible.
</details>

<details><summary><b>5.</b> What does longest-prefix-first scheduling risk, and how do you bound it?</summary>

Starving requests with no cached prefix. Combine it with aging (P6.2) so every request's wait is bounded.
</details>
