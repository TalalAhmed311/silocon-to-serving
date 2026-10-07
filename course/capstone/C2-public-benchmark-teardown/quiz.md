# C2 quiz

<details><summary><b>1.</b> Why does the report builder render TODO instead of an estimate when a result is missing?</summary>

An estimate in a benchmark table is indistinguishable from a measurement once it's screenshotted. Missing data must look missing.
</details>

<details><summary><b>2.</b> What must accompany a throughput number for it to be reproducible?</summary>

Hardware (instance, GPU, driver/CUDA), model and dtype, engine version and flags (batch limits, context length), workload definition (arrival process, prompt/output lengths, seed), SLOs used for goodput, and the number of runs with their spread.
</details>

<details><summary><b>3.</b> Your engine is 8 % slower than vLLM; run-to-run spread is ±6 %. What can you claim?</summary>

That the two are within noise of each other on this setup, unless more runs tighten the interval. Not that vLLM is 8 % faster.
</details>
