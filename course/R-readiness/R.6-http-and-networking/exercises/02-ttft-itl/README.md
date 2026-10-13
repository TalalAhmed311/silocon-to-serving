# Exercise 2: TTFT and inter-token latency

Implement `stream_metrics(request_sent, arrivals)` in `latency.py`. TTFT is the time from sending the request to the first chunk. The inter-token latency (ITL) values are the gaps between consecutive chunks; report their mean and 90th percentile (numpy's default linear method). End-to-end time is the last arrival minus the send time. P1.3 builds on these definitions.

```bash
uv run pytest course/R-readiness/R.6-http-and-networking/exercises/02-ttft-itl
```
