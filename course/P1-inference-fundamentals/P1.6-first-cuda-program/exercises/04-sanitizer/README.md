# Exercise 4: catch an out-of-bounds write with compute-sanitizer (T2, easy, no test)

`oob.cu` is a vector-scale program with the bounds check removed. It launches 4 blocks of 256 threads for 1,000 elements, so threads 1,000 to 1,023 write 24 floats past the end of the buffer. The program prints the right answer and exits normally. The CUDA runtime does not notice: allocations are usually padded, and the stray writes tend to land in the padding. A test would probably pass. The bug is still real: change the allocation or the size and it can corrupt a neighbouring buffer or faults.

## Run it

```bash
cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/p1.6 -j --target p1.6_04-sanitizer-oob          # built with -lineinfo, so reports carry source lines
./build/p1.6/p1.6_04-sanitizer-oob                                  # prints "h[0] = 2, h[999] = 2": looks fine
compute-sanitizer ./build/p1.6/p1.6_04-sanitizer-oob                # TODO(run-on: T2): the real report
compute-sanitizer --tool memcheck --show-backtrace=no ./build/p1.6/p1.6_04-sanitizer-oob 2>&1 | head -30
```

`compute-sanitizer` runs `memcheck` by default. It watches every global-memory access against the exact bounds of each allocation, so it sees what the padding hides. It makes the program much slower, so use it on small inputs. Docs: <https://docs.nvidia.com/compute-sanitizer/>.

## What the report looks like

The shape below is written by hand to show how to read one. **It is not real output.** Addresses, kernel signature text and the number of errors differ on your machine. Paste your real first error here (`TODO(run-on: T2)`).

```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at scale_kernel(float *, int, float)+0x...
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x... is out of bounds
=========     and is 4 bytes after the nearest allocation at 0x... of size 4000 bytes
=========     ... (more lines, one block per bad thread or per warp)
========= ERROR SUMMARY: ... errors
```

## Your task

1. Run it and copy the **first** error block into a comment at the end of `oob.cu`.
2. Explain each part in your own words: which kind of access, which thread and block, how far past the end, and the allocation size. Check one number by hand: block 3, thread 232 is global index 3 × 256 + 232 = 1000, the first element that does not exist.
3. Fix the kernel (`if (i < n)`), rebuild, and confirm the sanitizer reports no errors.
4. Optional: run `compute-sanitizer --tool racecheck` and `--tool initcheck` on it. What do those tools look for? (You will need `racecheck` in P5.2 when you meet shared memory.)

**Why not always use it?** It is a debugging tool, not a test harness: slow, and it only sees the inputs you give it. Use it whenever a kernel gives wrong answers, or intermittently wrong answers, or after any change to indexing.
