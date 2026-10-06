# Exercise 3 — Eager vs CUDA graphs at batch 1 (T2)

Run `bench/decode_ceiling.py` twice: with the default server (CUDA graphs), and with `01_serve.sh … --enforce-eager`. Record both batch-1 tok/s values in `results/p21.json`.

**Questions:**

1. Which is faster at batch 1, and by how much? Why does the gap shrink at batch 8? Think about launch overhead per step vs kernel time per step.
2. Graph capture costs start-up time and memory. Find both in `vllm.log`.
3. P2.6 and P6.6 go into what is captured. For now, predict whether the gap grows or shrinks for a 1B model vs an 8B model, and why.
