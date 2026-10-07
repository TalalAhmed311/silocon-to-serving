# Exercise 4: paged decode attention on the GPU (T2)

The kernel is P5.8's exercise 4 ([`../../../P5-cuda-deep-dive/P5.8-flashattention/exercises/04-paged-decode/README.md`](../../../P5-cuda-deep-dive/P5.8-flashattention/exercises/04-paged-decode/README.md)); its production version lives in D4 (`platform/kernels/include/d4/attention.cuh`, tested by `platform/kernels/tests/test_attention.cu`). Here you connect it to the engine:

1. Build the inputs exactly as `TorchRunner.decode` does: `block_tables [B, max_blocks]` (padded), `context_lens [B]`, and the paged pool `[num_blocks · block_size, kv_heads, head_dim]`.
2. Replace the gather-based attention inside `TorchRunner.decode` with a call to your kernel through the `torch_ext` pattern from P5.10 (`platform/kernels/torch_ext/`).
3. Check `tests/test_torch_runner.py` still passes, then compare decode step time, gather vs kernel, at B = 1, 8, 32 and context 512, 2048.

| B | context | gather (ms/step) | paged kernel (ms/step) |
|---|---|---|---|
| 1 | 512 | `TODO(run-on: g6.xlarge)` | |
| 8 | 2048 | | |
| 32 | 2048 | | |

Why does the gather version get *worse* relative to the kernel as the context grows? (Hint: it materialises a `[B, S, kv_heads, hd]` copy every layer.)
