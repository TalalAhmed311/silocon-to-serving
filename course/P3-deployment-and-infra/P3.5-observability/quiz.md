# P3.5 quiz

<details><summary><b>1.</b> Why alert on burn rate rather than on "p90 TTFT > 2 s"?</summary>

Burn rate ties the alert to user impact against a budget, fires fast for severe problems and slowly for leaks, and resets quickly after a fix (the short window). A raw-threshold alert is noisy and ignores duration.
</details>

<details><summary><b>2.</b> Why is <code>avg by (pod)</code> of per-pod p90s wrong?</summary>

Percentiles don't average. Sum the histogram buckets across pods first, then call `histogram_quantile`.
</details>

<details><summary><b>3.</b> DCGM says the GPU is 100% utilized. Is the GPU efficient?</summary>

Not necessarily. GPU_UTIL is "a kernel was running". Memory-bound decode can show 100% while the tensor cores are mostly idle. Compare tokens/s against your measured capacity.
</details>

<details><summary><b>4.</b> Why report idle capacity as separate "waste" rather than spreading it across tenants?</summary>

It makes the real lever visible (utilization, autoscaling, packing), and keeps per-tenant rates comparable and fair.
</details>

<details><summary><b>5.</b> Where does a request ID belong: a metric label, a trace attribute, or a log field?</summary>

A trace attribute or log field. As a metric label it explodes cardinality.
</details>
