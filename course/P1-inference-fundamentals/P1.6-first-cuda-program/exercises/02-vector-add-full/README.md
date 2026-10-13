# Exercise 2: the full vector add (T2, medium)

Write the complete host program for `c = a + b` in `kernel.cuh`: the kernel, then the five steps (allocate, copy in, launch, copy out, free) with every CUDA call checked. The function takes **host** pointers, like a normal C++ function, and hides the GPU inside.

Use `ceil-div` for the grid and a bounds check in the kernel. The test runs N = 1, 1000, 2²⁰ and 2²⁰ + 3 and compares with a CPU loop. The last size leaves 253 idle threads in the final block (4,097 blocks × 256 = 1,048,832 threads for 1,048,579 elements).

```bash
cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/p1.6 -j
ctest --test-dir build/p1.6 -R p1.6_02-vector-add-full --output-on-failure      # TODO(run-on: T2)
```

**Questions.** Delete the `if (i < n)` guard and run again. Does the test fail? Now run it under `compute-sanitizer` (exercise 4 shows how). Which is the better detector, the test or the tool? Why must `cudaMalloc` take `&d_a` and not `d_a`?
