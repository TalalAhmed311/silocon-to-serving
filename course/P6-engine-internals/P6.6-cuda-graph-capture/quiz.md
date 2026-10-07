# P6.6 quiz

<details><summary><b>1.</b> What does a CUDA graph save, and when does it matter?</summary>

The per-kernel CPU launch cost (Python, dispatcher, driver). It matters when kernels are short enough that the CPU can't keep the GPU fed — small-batch decode.
</details>

<details><summary><b>2.</b> Why must inputs be copied into static buffers before replay?</summary>

The graph recorded the device addresses of the tensors used at capture; replay reads those addresses, not whatever tensor your Python variable now names.
</details>

<details><summary><b>3.</b> The running batch has 5 sequences; buckets are 1, 2, 4, 8. What happens?</summary>

The batch is padded to 8 and the 8-graph replays; the 3 padded rows compute garbage that is ignored (and write to a reserved slot).
</details>

<details><summary><b>4.</b> Why is prefill usually left eager?</summary>

Its shapes vary with every chunk (number of tokens, context lengths), and its kernels are large enough that launch overhead is a small fraction of the step.
</details>

<details><summary><b>5.</b> Name two ways a host sync sneaks into a model's forward pass.</summary>

`.item()` / `.tolist()` / `.cpu()` on a tensor (often to size a buffer or branch), and Python control flow on tensor values (`if x.any():`). Printing a CUDA tensor too.
</details>
