# Exercise 4: cross-check against the Modular handbook calculator

Open *Calculating GPU memory for LLMs* in the Modular handbook (`docs/getting-started/calculating-gpu-memory-for-llms.md` in `modular/llm-inference-handbook@5bddc17b`; the rendered page has the interactive calculator). For each row, enter the same model, precision and context in both tools.

| model | GPU | weights / KV dtype | context × batch | handbook total GB | our weights + KV GB | difference | why |
|---|---|---|---|---|---|---|---|
| llama3-8b | L4 | bf16 / bf16 | 4096 × 8 | | | | |
| llama3-70b | H100-SXM ×4 | fp8 / fp8 | 8192 × 32 | | | | |
| mixtral-8x7b | L40S ×2 | bf16 / bf16 | 4096 × 16 | | | | |

Typical reasons for a difference:
- overhead or activation estimates
- GB vs GiB
- a different KV formula (MHA vs GQA)
- MoE handling
- whether the LM head is counted
