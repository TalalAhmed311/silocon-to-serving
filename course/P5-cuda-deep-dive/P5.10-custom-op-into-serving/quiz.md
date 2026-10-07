# P5.10 quiz

<details><summary><b>1.</b> Why must `mutates_args` list `x` and `residual`?</summary>

The compiler (functionalization, reordering, dead-code elimination) relies on the schema. An undeclared in-place write can be reordered or dropped, giving silently wrong results.
</details>

<details><summary><b>2.</b> What does the fake implementation do, and why does torch.compile need it?</summary>

It describes outputs (shapes, dtypes, strides) without computing them. Dynamo and Inductor use it to trace and plan memory without running the real kernel.
</details>

<details><summary><b>3.</b> Why patch through a vLLM plugin rather than in your launch script?</summary>

vLLM runs the model in separate processes (engine core and workers). Plugins load in every process, while a patch in the launcher process never reaches the model.
</details>

<details><summary><b>4.</b> Your kernel is 2× faster than vLLM's and RMSNorm is 3% of a decode step. Expected end-to-end gain?</summary>

About 1.5% of step time, likely within run-to-run noise. Measure several runs and report the spread.
</details>
