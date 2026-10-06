# P2.2 quiz

<details><summary><b>1.</b> What is SGLang's equivalent of vLLM's <code>--gpu-memory-utilization</code>?</summary>

`--mem-fraction-static` (verify it at your pinned version, exercise 1).
</details>

<details><summary><b>2.</b> Why does a radix-tree prefix cache help multi-turn chat so much?</summary>

Each turn resends the whole history. The cache matches the longest previously-computed prefix, so only the new tokens need prefill.
</details>

<details><summary><b>3.</b> Why shouldn't you install vLLM and SGLang in one venv?</summary>

Each pins specific torch, flashinfer and CUDA builds. Mixing them breaks one or both.
</details>

<details><summary><b>4.</b> What does a longest-prefix-match scheduling policy trade?</summary>

It reorders the waiting queue to maximize cache hits, which raises throughput, at some cost in fairness: requests with no shared prefix can wait longer.
</details>
