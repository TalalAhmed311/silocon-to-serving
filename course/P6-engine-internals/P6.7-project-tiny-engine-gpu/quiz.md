# P6.7 quiz

<details><summary><b>1.</b> Which single test gives the most confidence that paging, chunking and preemption are correct, and why?</summary>

The paged engine vs the contiguous v0 reference, greedy, token for token, under a small budget and a small pool: any wrong slot, mask, position or recompute changes a token.
</details>

<details><summary><b>2.</b> Your engine reaches 40 % of vLLM's tok/s at batch 32. Name the first three things your profile should tell you.</summary>

The fraction of the step that is GPU-idle (CPU/launch-bound vs GPU-bound); the top kernels by time and how close each is to its roofline; and whether attention (gather vs paged kernel) or GEMMs dominate.
</details>

<details><summary><b>3.</b> Why must the comparison fix max_num_seqs and max_num_batched_tokens on both engines?</summary>

They bound the batch size and prefill chunking, which set the throughput/latency trade-off. With different limits you compare policies, not implementations.
</details>

<details><summary><b>4.</b> What does the engine need from the GPU runner per step, and what does it return?</summary>

Input: the scheduled chunks (sequence, start, token count) with block tables and pending CoW copies. Output: logits for every chunk that produces a token.
</details>
