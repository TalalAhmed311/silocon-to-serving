# P2.3 quiz

<details><summary><b>1.</b> What is wasted in static batching that continuous batching recovers?</summary>

The slots of sequences that finished early (they idle until the longest finishes), and the time new arrivals spend waiting for the whole batch to end.
</details>

<details><summary><b>2.</b> Why does chunked prefill lower worst-case ITL but raise TTFT for long prompts?</summary>

Each step's cost is capped at the chunk size, so decodes are never stuck behind a huge prefill. The long prompt's prefill is spread over several steps, so its first token comes later.
</details>

<details><summary><b>3.</b> What does a PagedAttention block table map, and why does it nearly eliminate fragmentation?</summary>

It maps a sequence's logical block index to a physical KV block. Blocks are allocated on demand in small fixed sizes, so waste is under one block per sequence instead of max_model_len minus the actual length.
</details>

<details><summary><b>4.</b> Two prompts share 1,000 tokens, then differ. With 16-token blocks, how many blocks can be shared?</summary>

⌊1000/16⌋ = **62** full blocks. The partial 63rd block isn't hashed or shared.
</details>

<details><summary><b>5.</b> Why must a load generator be open-loop to find the knee?</summary>

A closed-loop client slows down when the server does, so the offered load never exceeds capacity and the queue never builds (coordinated omission).
</details>

<details><summary><b>6.</b> Throughput flattens and <code>vllm:num_preemptions</code> rises. What is the bottleneck?</summary>

KV-cache capacity. The scheduler evicts running sequences to admit new ones and recomputes them later, which wastes work. Remedies: FP8 KV, a lower `max_num_seqs`, more GPUs, or shorter `max_model_len`.
</details>
