# Exercise 1: hello, device (T2, easy)

Write your first kernel in `kernel.cuh`. Launch **2 blocks of 4 threads**. Each thread computes its global index, prints one line with `printf`, and stores the index in `out[i]`.

Expected output (the order of lines varies from run to run: threads are not ordered):

```
block 1 thread 0 -> global 4
block 0 thread 0 -> global 0
...
```

Run (`TODO(run-on: T2)`):

```bash
cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/p1.6 -j
ctest --test-dir build/p1.6 -R p1.6_01-hello-device --output-on-failure
```

**Questions.** Is the output order the same on every run? Why does the test call `CUDA_CHECK_LAUNCH()` before reading the result? What would you see if you removed it and the program ended right after the launch?
