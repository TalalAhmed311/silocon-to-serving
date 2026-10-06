# P1.3 quiz

<details><summary><b>1.</b> A request arrives at t = 0, its first token comes at 0.4 s and its 11th (last) at 1.4 s. TTFT, TPOT, E2E?</summary>

TTFT 0.4 s. TPOT (1.4 − 0.4)/10 = **0.1 s**. E2E 1.4 s.
</details>

<details><summary><b>2.</b> Why report p90/p99 instead of the mean?</summary>

Latency distributions are right-skewed. A few queued requests inflate the mean without telling you how often users suffer. Percentiles describe the experience directly.
</details>

<details><summary><b>3.</b> What is goodput, and why is it a better optimization target than throughput?</summary>

It is requests per second that met the SLO. Pushing throughput past the knee yields requests that are fast in aggregate but too slow individually, so goodput falls even as throughput rises.
</details>

<details><summary><b>4.</b> Prometheus buckets are …0.5, 1, 2.5… and the true p90 is 0.6 s. What can <code>histogram_quantile</code> report?</summary>

Anything in (0.5, 1]. It interpolates linearly inside the bucket, so the precision is the bucket width.
</details>

<details><summary><b>5.</b> Which metric explodes first as load approaches capacity, TTFT or ITL? Why?</summary>

**TTFT.** The waiting queue grows without bound past capacity, and every queued second adds to TTFT. ITL only grows with the batch size, which is capped by `max_num_seqs`.
</details>

<details><summary><b>6.</b> Client-measured TTFT is consistently 30 ms above the server's. What's in those 30 ms?</summary>

Network round trip, TLS, proxies and gateways, the server's accept and HTTP parsing, and response buffering. It is worth tracking separately.
</details>
