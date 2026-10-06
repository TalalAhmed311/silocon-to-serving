# P2.1 quiz

<details><summary><b>1.</b> Which flag decides how many tokens of KV cache vLLM allocates?</summary>

`--gpu-memory-utilization`. KV = utilization × VRAM − weights − profiled activation memory. `--max-model-len` only caps one sequence and checks it fits.
</details>

<details><summary><b>2.</b> Why bind to 127.0.0.1 and use an SSM port-forward instead of opening port 8000?</summary>

So no unauthenticated, publicly reachable inference endpoint ever exists. The port-forward is authenticated by your AWS identity.
</details>

<details><summary><b>3.</b> Batch-1 decode is 75% of the bandwidth ceiling. Name three contributors to the missing 25%.</summary>

Achievable vs peak bandwidth, KV-cache reads, kernel launch gaps and non-GEMM ops, sampling, detokenization and HTTP.
</details>

<details><summary><b>4.</b> Why does aggregate tok/s rise almost linearly from batch 1 to 8 at short context?</summary>

Decode is memory-bound and the weight reads are shared across the batch. The step time barely changes while 8× more tokens come out.
</details>

<details><summary><b>5.</b> When would you use the offline <code>LLM.generate</code> API instead of the server?</summary>

For batch jobs (evals, synthetic data, embeddings) where you have all prompts up front and care about throughput, not latency.
</details>
