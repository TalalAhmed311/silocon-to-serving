# Exercise 4 — Disaggregated vs aggregated (T3, 2 GPUs)

Follow `platform/deploy/dynamo/README.md` (or llm-d's disaggregation guide). Use the same 2 GPUs in both setups: 2 aggregated replicas, then 1 prefill + 1 decode. Use a long-prompt workload: `loadgen.cli --prompt-mean 4000 --output-mean 200 --rates 0.5 1 2 4`.

| setup | rate | p50 TTFT | p90 TTFT | p50 ITL | p90 ITL | goodput |
|---|---|---|---|---|---|---|

Explain the result with §2's arithmetic: KV bytes per request, your measured network bandwidth between the nodes (`iperf3` pod to pod), and your measured prefill time. Say at what rate and prompt length disaggregation started (or would start) to pay on *this* network.
