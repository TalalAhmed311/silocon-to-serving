# P1.6 quiz

<details><summary><b>1.</b> You call <code>cudaMalloc(&amp;d_x, bytes)</code> and then write <code>d_x[0] = 1.0f;</code> in host code. What happens, and why?</summary>

It crashes or corrupts memory. `d_x` holds an address in **GPU** memory. The CPU cannot read or write it through an ordinary pointer. Data has to be moved with `cudaMemcpy` (or a kernel writes it).
</details>

<details><summary><b>2.</b> Put the five steps in order: copy result back, launch, free, allocate device memory, copy inputs in.</summary>

Allocate, copy in, launch, copy back, free. The result copy comes after the launch because the kernel produces it, and the free comes last because the copies and kernel use the buffers.
</details>

<details><summary><b>3.</b> N = 1,000,000 and 256 threads per block. How many blocks, and how many threads are switched off by the bounds check?</summary>

ceil(1,000,000 ÷ 256) = (1,000,000 + 255) ÷ 256 = 3,907 blocks (3,906.25 rounds up). That is 3,907 × 256 = 1,000,192 threads, so 192 threads have no element and `if (i < n)` switches them off.
</details>

<details><summary><b>4.</b> Your kernel launch asks for 2,048 threads per block. Which call reports it, and when?</summary>

`cudaGetLastError()` right after the launch, returning `cudaErrorInvalidConfiguration`. The limit is 1,024 threads per block, and a bad configuration is detected when the launch is made. Without the check, no thread runs and nothing is reported.
</details>

<details><summary><b>5.</b> A kernel writes out of bounds. <code>cudaGetLastError()</code> right after the launch returns success. Why, and where does the error show up?</summary>

The launch itself was valid and has only been queued. The write happens later, on the GPU. The error is reported by the next call that waits for the GPU (`cudaDeviceSynchronize()`, or a `cudaMemcpy`), which can be far from the buggy kernel. Run under `compute-sanitizer`, or set `CUDA_LAUNCH_BLOCKING=1`, to find the real cause.
</details>

<details><summary><b>6.</b> Why is timing a kernel with <code>std::chrono</code> around the launch wrong?</summary>

The launch returns as soon as the work is queued, so the host clock measures the queuing, not the kernel. Use CUDA events recorded on the GPU's timeline, and synchronize on the end event before reading the time. Also exclude the first call, which pays for context creation.
</details>

<details><summary><b>7.</b> A vector add of 2²⁶ floats takes 2.5 ms in the kernel but the whole program takes about 70 ms. Where is the time, and what does it suggest?</summary>

In the copies: two input arrays in and one result out cross PCIe, which is far slower than GPU memory (about 805 MB at an example 12 GB/s is roughly 67 ms; the kernel's 805 MB at 320 GB/s is about 2.5 ms). Moving to the GPU only pays off when the data stays there for many kernel passes. The crossover is about memory bandwidth ÷ PCIe bandwidth passes (here about 27). This is why inference engines keep weights and the KV cache resident on the GPU. (Both bandwidths are example values.)
</details>

<details><summary><b>8.</b> <code>compute-sanitizer</code> reports an invalid write, but your program printed the right answer. Is there a bug?</summary>

Yes. The hardware did not fault because the stray write landed in memory the allocator had padded, or in a neighbouring allocation that was not read afterwards. Correct output on one run proves nothing. The sanitizer checks exact allocation bounds, so trust it and fix the missing `if (i < n)`.
</details>
