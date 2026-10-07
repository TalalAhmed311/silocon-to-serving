# P6.2 quiz

<details><summary><b>1.</b> Why are running sequences scheduled before waiting ones?</summary>

They already hold KV memory and users are watching their streams; starving them to admit newcomers raises ITL and wastes the memory they hold. Admitting first also invites thrash (admit → preempt → readmit).
</details>

<details><summary><b>2.</b> A prefill chunk covers prompt positions 100–163 of a 400-token prompt. Does the engine sample a token after it?</summary>

No. Only a chunk that reaches the end of the sequence's known tokens produces logits that matter for the next token.
</details>

<details><summary><b>3.</b> Recompute or swap on a GPU with fast compute and a slow PCIe link?</summary>

Recompute: re-prefilling is compute-bound and fast on the GPU, while swap moves the KV over PCIe twice.
</details>

<details><summary><b>4.</b> Why can't the victim be a sequence already placed in this step's batch?</summary>

Its blocks would be freed (and possibly handed to another sequence) while the forward pass is about to write its K/V into them.
</details>

<details><summary><b>5.</b> What does aging buy, and what does it cost?</summary>

A bounded worst-case wait for low-priority work. The cost is weaker priority isolation: high-priority requests occasionally wait behind an aged low-priority one.
</details>
