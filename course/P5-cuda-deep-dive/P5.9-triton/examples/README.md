# P5.9 examples

| File | What |
|---|---|
| [`platform/kernels/triton_kernels/`](../../../../platform/kernels/triton_kernels/__init__.py) | the reference Triton rewrites: `softmax.py`, `rmsnorm.py`, `matmul.py`, `flash_attention.py` |
| [`bench_triton_vs_cuda.py`](bench_triton_vs_cuda.py) | Triton vs PyTorch on the D4 shapes, next to your CUDA numbers |
| Triton `v3.8.0` tutorials `01-vector-add`, `02-fused-softmax`, `03-matrix-multiplication`, `05-layer-norm`, `06-fused-attention` | run and annotate them on the GPU box (linked, not copied): `python/tutorials/` in triton-lang/triton at the pinned tag |
| Triton Puzzles (srush/Triton-Puzzles) | exercise 5. Runs in the interpreter, no GPU needed |
