# P6.1 quiz

<details><summary><b>1.</b> Name the three decisions an engine makes every step and the component that owns each.</summary>

What to compute (scheduler), where its KV lives (block manager / KV cache manager), and which token comes next (sampler). The model runner just executes the first decision.
</details>

<details><summary><b>2.</b> Why is a block's prefix hash chained on the previous block's hash?</summary>

Because a block's KV depends on every earlier token, not just its own. Two blocks with identical tokens but different prefixes have different K/V; chaining makes their hashes differ.
</details>

<details><summary><b>3.</b> After a prefix-cache hit of 32 tokens on a 40-token prompt, what is the sequence's computed-token count, and what is scheduled next?</summary>

32. The scheduler schedules the remaining 8 prompt tokens (or a chunk of them); the last prompt token always runs through the model so the engine gets logits for the first output token.
</details>

<details><summary><b>4.</b> What does the model runner need from the scheduler to run a mixed prefill+decode batch?</summary>

For every scheduled token: its id, its position, and the cache slot to write its K/V (slot mapping). For attention: each sequence's block table and context length (and query start offsets for varlen prefill).
</details>

<details><summary><b>5.</b> Why can CPU-side scheduling dominate a decode step at batch 1?</summary>

A small-batch decode step is a few milliseconds of memory-bound GPU work split across many small kernels; Python scheduling, input preparation and per-kernel launch overhead are a comparable fixed cost. CUDA graphs (P6.6) and async scheduling hide it.
</details>
