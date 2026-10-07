# P5.9 quiz

<details><summary><b>1.</b> In Triton, what does one program correspond to in CUDA terms?</summary>

Roughly one thread block, but you write operations on whole tensors (blocks) rather than per-thread code. The compiler maps them to warps and threads.
</details>

<details><summary><b>2.</b> Why does `tl.load(..., mask=m, other=-inf)` matter in softmax?</summary>

Masked lanes otherwise read garbage or zeros. With −inf they don't affect the max, and exp(−inf) = 0 doesn't affect the sum.
</details>

<details><summary><b>3.</b> What does grouped program ordering in matmul improve?</summary>

L2 hit rate. Programs running at the same time reuse the same A row panels and B column panels instead of streaming the whole matrix.
</details>

<details><summary><b>4.</b> Where did `cp.async` multi-stage pipelining go?</summary>

Into `num_stages` in the autotune config. The compiler generates the async copies and the stage bookkeeping.
</details>

<details><summary><b>5.</b> How can Triton kernels be tested in CI without a GPU?</summary>

With the interpreter (`TRITON_INTERPRET=1`), which executes the block operations in Python/NumPy on CPU tensors. It checks correctness, not performance.
</details>
