# P2.4 quiz

<details><summary><b>1.</b> What is coordinated omission?</summary>

A closed-loop load generator waits for slow responses before sending more, so it omits the requests that would have arrived during the slowdown. Latency percentiles then look better than real users experience.
</details>

<details><summary><b>2.</b> You have 150 samples. Can you report p99?</summary>

Only barely: it is about the 2nd-largest value. Report it with n, and prefer p90, or collect ≥ 1000 samples.
</details>

<details><summary><b>3.</b> Why can't you compare <code>vllm bench throughput</code> with a serving benchmark?</summary>

It runs offline, with no HTTP, all prompts available up front and maximal batching. It measures engine capacity, not serving latency under arrivals.
</details>

<details><summary><b>4.</b> Name five fields a reproducible LLM benchmark report must include.</summary>

Any five of: hardware (GPU, count, driver), engine and version, model and revision and dtype, non-default flags, load model and rate, seed, prompt/output distributions and tokenizer, n and repeats, the raw data.
</details>
