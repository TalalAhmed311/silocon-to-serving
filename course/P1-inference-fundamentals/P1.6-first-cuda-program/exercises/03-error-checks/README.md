# Exercise 3: make the errors speak (T2, medium)

`kernel.cuh` has two planted bugs. Neither crashes. Both return `cudaSuccess` and the caller gets garbage or nothing:

1. `launch_scale(d_x, n, 2048)` asks for 2,048 threads per block. The limit is 1,024, so the launch is rejected and **no thread runs**. Nothing tells you.
2. `launch_fault()` launches a valid configuration, but the kernel writes through a null pointer. The fault happens later, on the GPU, after the launch has already returned.

Add the checks so each function returns the right `cudaError_t`: `cudaGetLastError()` right after the launch (a bad configuration is known at once) and `cudaDeviceSynchronize()` (a fault is known only when the kernel has run). Do **not** change the 2048: the test expects it to be reported.

```bash
cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/p1.6 -j
ctest --test-dir build/p1.6 -R p1.6_03-error-checks --output-on-failure      # TODO(run-on: T2)
```

**Questions.** Why does the test with a valid launch pass on the starter even though the starter never synchronizes? (Hint: what does `cudaMemcpy` on the default stream wait for?) Why is the faulting case last in the test file? What does `CUDA_LAUNCH_BLOCKING=1` change?
