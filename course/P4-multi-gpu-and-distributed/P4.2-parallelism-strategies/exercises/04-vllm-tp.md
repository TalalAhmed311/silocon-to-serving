# Exercise 4 (hard): measured vLLM TP = 1/2/4 vs the prediction (T3)

Run `examples/03_vllm_tp.sh` on the 4× L4 box (see [aws.md](../aws.md)). For each TP:

1. batch-1 ITL from the lowest-rate run (`results/p4.2/tp*.json`), next to D3's prediction from exercise 3
2. the #4 knee in output tok/s, and tok/s **per GPU**
3. from the vLLM startup log: weights GB per GPU and KV-cache capacity (tokens)

Write 5–10 sentences explaining every gap over 20% between prediction and measurement. Candidates: CUDA graphs on or off, the custom all-reduce kernel vs NCCL on PCIe, the KV capacity difference changing the max batch, CPU overhead per step, and kernel efficiency at smaller per-GPU matrix sizes.

Deliverable: the filled bench table plus the write-up. `TODO(run-on: g6.12xlarge)`
