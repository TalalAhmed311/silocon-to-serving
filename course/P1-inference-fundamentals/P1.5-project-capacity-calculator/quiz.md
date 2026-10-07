# P1.5 quiz

<details><summary><b>1.</b> Which three inputs move "max sequences" the most?</summary>

KV dtype (bytes per token), context length, and weight dtype (it frees memory for KV). TP is next: it splits both.
</details>

<details><summary><b>2.</b> Why does the calculator use <i>active</i> params for decode speed but <i>total</i> params for VRAM on MoE models?</summary>

All experts must be resident in memory, but each token reads only the routed experts' MLP weights.
</details>

<details><summary><b>3.</b> vLLM reports 20% fewer KV tokens than your prediction. List three candidate causes.</summary>

The activation/profiling reserve is larger than assumed, CUDA graph memory, a different `gpu_memory_utilization`, block-size rounding, or extra tensors (for example LoRA or spec-decode drafts).
</details>

<details><summary><b>4.</b> Why is the prefill estimate a <i>floor</i> on TTFT?</summary>

It ignores queueing and assumes the GPU does nothing else. Under load, prefill shares steps with other requests' decodes.
</details>
