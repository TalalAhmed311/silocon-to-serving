# P5.9 exercises

Implement the four files in [`ours/`](ours/). `test_triton.py` tests them on a GPU if there is one, and otherwise in Triton's **interpreter** on CPU tensors (`TRITON_INTERPRET=1`, set automatically): the T0 path, also used in CI. `S2S_SOLUTIONS=1` runs the same tests against `platform/kernels/triton_kernels/`.

| # | File | Test | Then compare with |
|---|---|---|---|
| 1 | `ours/softmax.py`: one program per row, row in one block; an online loop for rows longer than one block | `test_softmax` (incl. 20,000-column rows) | your P5.5 CUDA softmax: GB/s |
| 2 | `ours/rmsnorm.py`: fp32 accumulation, optional fused residual updated in place | `test_rmsnorm` (fp32 and bf16; residual exact) | P5.5 `fused_add_rmsnorm` |
| 3 | `ours/matmul.py`: block tiles, `tl.dot`, masks for any shape, grouped ordering, autotune list | `test_matmul` (odd shapes, fp32 + fp16) | your P5.6 rung 7 (fp32) and P5.7 HGEMM (fp16) |
| 4 | `ours/flash_attention.py`: FA-2 forward, causal | `test_flash_attention` | your P5.8 CUDA FA-2 |
| 5 | *(hard)* Triton Puzzles: puzzles 1–8 at least | the puzzles' own checks | — |

```bash
uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch      # T0: interpreter
uv run python course/P5-cuda-deep-dive/P5.9-triton/examples/bench_triton_vs_cuda.py      # T2
```

Deliverable: the CUDA vs Triton table (README bench), plus 3–5 sentences per kernel on what Triton did for you (tiling, shared memory, tensor-core layouts, pipelining via `num_stages`) and what you couldn't control.
