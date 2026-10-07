# 85: LoRA Linear (L5, practice)

Problem: [LeetGPU #85](../../leetgpu-map.md). `y = x·W + (α/r)·x·A·B` with a low-rank adapter (r ≪ K, N).

**Hint ladder**

1. Associativity matters: `(x·A)·B` costs `M·K·r + M·r·N` FLOPs, while `x·(A·B)` materializes a K×N matrix first. Always multiply by A first.
2. `t = x·A` is a skinny GEMM (M×r, with r = 8–64). Then do the main GEMM `x·W` and add `(α/r)·t·B` **in its epilogue**: each output needs r extra FMAs, and no M×N temporary ever hits HBM.
3. Serving many adapters at once (S-LoRA/Punica style) batches requests that use different A, B: a "segmented" GEMM where each row picks its adapter by index.

**Solution outline:** `gemm_plain` for `t`, then `lora_gemm` = the #22 tile core + an epilogue loop over r.

**Why it matters:** multi-LoRA serving (one base model, hundreds of fine-tunes) depends on that fused epilogue and on batching across adapters.
