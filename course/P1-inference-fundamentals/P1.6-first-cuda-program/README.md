# P1.6: Your first CUDA program

| | |
|---|---|
| **Tier** | ![T1](https://img.shields.io/badge/tier-T1%20browser%20GPU-blue) for the kernel (LeetGPU or Colab) · ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for the full host program (`g4dn.xlarge`, see [aws.md](aws.md)) |
| **Time** | ≈30 min reading + ≈5 h hands-on |
| **Prerequisites** | [P1.4](../P1.4-gpu-architecture-and-roofline/README.md) (what an SM, a warp and GPU memory are), [R.1](../../R-readiness/R.1-cpp-from-zero/README.md) and [R.2](../../R-readiness/R.2-memory-management/README.md) (pointers, heap, ownership). Tools: a C++ compiler and, on the GPU box, CUDA 12.4+ and CMake 3.24+ |
| **You will build** | 4 exercises: a hello-world kernel, a full vector-add host program, error checks, a sanitizer run; 2 examples that time your GPU |
| **Reading** | NVIDIA blog, *An Even Easier Introduction to CUDA* · CUDA C++ Programming Guide, *Programming Model* (links in Go deeper) |

## Learning objectives

By the end of this module you can:

1. Explain the **host/device split**: two processors, two memories, one slow link (PCIe), and why a pointer from `cudaMalloc` is meaningless on the CPU.
2. Get a GPU (LeetGPU, Colab or AWS), check it with `nvidia-smi`, and compile a `.cu` file with `nvcc` for the right architecture (`-arch=sm_75`).
3. Write a `__global__` kernel that finds its element with `blockIdx.x * blockDim.x + threadIdx.x`, with a bounds check.
4. Write the **five-step host program**: allocate, copy in, launch, copy out, free. Size the grid with ceil-div for any N.
5. Say why launches are asynchronous, and check errors in the right places with `CUDA_CHECK` and `CUDA_CHECK_LAUNCH()`.
6. Debug with `printf` in a kernel and `compute-sanitizer`.
7. Time a GPU program honestly with CUDA events, and explain why the copies can cost more than the kernel.

## Why this sits here

P1.4 told you what a GPU is. This module makes you write for one, on purpose, before the course relies on it. [Lane B](../../../laneB-cuda/README.md) L1 (indexing in 1D, 2D and 3D, many problems) starts right after this and runs alongside P2 to P4. [P5](../../P5-cuda-deep-dive/syllabus.md), the performance deep dive, assumes this module and Lane B L1 to L3. It recaps both quickly and then goes deeper.

---

## 1. Host and device

The **host** is the CPU with its RAM. The **device** is the GPU with its own memory (on a T4, 16 GB of GDDR6 per [`gpu_specs.yaml`](../P1.4-gpu-architecture-and-roofline/gpu_specs.yaml), UNVERIFIED). They are connected by **PCIe**. The CPU is in charge: it allocates GPU memory, moves data, and tells the GPU what to run. The GPU runs only what it is given.

Two facts follow:

- A device pointer (from `cudaMalloc`) is an address in GPU memory. The CPU must never dereference it. It only passes it around, to `cudaMemcpy` and to kernels.
- PCIe is far slower than GPU memory. Example values, check your platform: PCIe 3.0 x16 is about 15.75 GB/s in theory (8 GT/s × 16 lanes × 128/130 ÷ 8) and often 10 to 13 GB/s measured. A T4's memory is about 320 GB/s (UNVERIFIED). Every byte that crosses PCIe costs about 25 times what a byte in GPU memory costs. That is the **transfer tax** (section 9).

## 2. Where to get a GPU

| Option | Tier | What it can do | Cost |
|---|---|---|---|
| [LeetGPU](https://leetgpu.com/) in the browser | T1 | Write and run a kernel. The site supplies the host side. No `nvcc` flags, no `cudaMalloc` | free tier: check the site |
| [Google Colab](https://colab.research.google.com/) with a GPU runtime | T1 | Full programs: write a `.cu` file in a cell and run `!nvcc ...`. GPU type and availability change: check what you get with `!nvidia-smi` | free tier: check current limits |
| AWS `g4dn.xlarge` (1× T4, sm_75) | T2 | Everything, including `compute-sanitizer` and Nsight. Set up exactly like P5: [aws.md](aws.md) | hours × hourly price (example value, check the price page); auto-stop and teardown in the shared guide |

First check on any machine:

```bash
nvidia-smi          # driver version, GPU name, memory; fails if there is no driver or no GPU
nvcc --version      # the CUDA toolkit (compiler); the driver and toolkit are separate installs
```

## 3. The toolchain

`nvcc` compiles `.cu` files. It splits each file in two: **host code** goes to your normal C++ compiler (g++), and **device code** (the `__global__` functions) is compiled to **PTX** (a virtual instruction set) and then to **SASS** (the real machine code of one GPU generation). Both end up in one executable.

```bash
nvcc -arch=sm_75 -lineinfo -I laneB-cuda/harness/include -o add add.cu
```

`sm_75` means *compute capability 7.5*: the version number of a GPU generation's feature set (T4 is 7.5). Build for the GPU you will run on, or CMake's `CMAKE_CUDA_ARCHITECTURES=native` picks it for you. The `examples/01_device_info.cu` program prints yours.

## 4. A kernel

```cpp
__global__ void hello(int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // my global index
  out[i] = i;
}
hello<<<2, 4>>>(d_out);                            // 2 blocks of 4 threads = 8 threads
```

`__global__` marks a function that the host launches and the device runs, once per thread. It returns `void`. `blockIdx.x` is which block, `blockDim.x` is the threads per block (4), `threadIdx.x` is the position inside the block. This module uses 1D only; L1 adds 2D and 3D.

## 5. The five-step program

```cpp
float *d_a, *d_b, *d_c;
size_t bytes = n * sizeof(float);
cudaMalloc(&d_a, bytes);  cudaMalloc(&d_b, bytes);  cudaMalloc(&d_c, bytes);       // 1 allocate
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);                                 // 2 copy in
cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
add_kernel<<<blocks, 256>>>(d_a, d_b, d_c, n);                                       // 3 launch
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);                                 // 4 copy out
cudaFree(d_a);  cudaFree(d_b);  cudaFree(d_c);                                       // 5 free
```

Details that trip everyone once: `cudaMemcpy(dst, src, bytes, kind)` has the destination first and counts **bytes**; `cudaMalloc` takes `&d_a` because it writes the new address into your pointer; and every `cudaMalloc` needs a `cudaFree` (R.2's ownership rules apply to the device too). This is [exercise 2](exercises/02-vector-add-full/README.md).

## 6. Ceil-div and the bounds check

N is rarely a multiple of the block size. Round the grid up, then switch off the extra threads:

```cpp
int blocks = (n + threads - 1) / threads;       // ceil(n / threads) in integer maths
// in the kernel:  if (i < n) c[i] = a[i] + b[i];
```

Example: n = 2²⁰ + 3 = 1,048,579 with 256 threads: 1,048,579 ÷ 256 = 4,096.01, so 4,097 blocks = 1,048,832 threads, and the last 253 must do nothing.

## 7. Launches are asynchronous

A launch queues work and returns at once. A bad **configuration** (say 2,048 threads per block, over the 1,024 limit) is known immediately: `cudaGetLastError()` returns it. A **fault inside the kernel** (an out-of-bounds write) is known only when the GPU gets there, and shows up at the next call that waits for the GPU, which may be a `cudaMemcpy` far from the cause.

```cpp
add_kernel<<<blocks, 256>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());        // was the launch valid?
CUDA_CHECK(cudaDeviceSynchronize());   // did the kernel run to the end?
// or, in one line:  CUDA_CHECK_LAUNCH();
```

`CUDA_CHECK(call)` and `CUDA_CHECK_LAUNCH()` live in [`laneB-cuda/harness/include/s2s_cuda.cuh`](../../../laneB-cuda/harness/include/s2s_cuda.cuh). `CUDA_CHECK` prints the error name with file and line and exits. Synchronizing after every launch is right for tests and debugging, and wrong for a tuned pipeline (P5.1 covers streams). `CUDA_LAUNCH_BLOCKING=1` in the environment makes every launch synchronous, so errors point at the right line.

## 8. Debugging

- **`printf` in a kernel** works, with caveats: output appears only when the GPU syncs, lines from different threads come in any order, and the buffer is a fixed size, so printing from millions of threads loses lines. Print from a few threads: `if (i < 4) printf(...)`.
- **`compute-sanitizer ./program`** runs the program with memory checking. It catches out-of-bounds and misaligned accesses that the hardware lets slide. Build with `-lineinfo` and the report names the source line. It is slow, so use small inputs. [Exercise 4](exercises/04-sanitizer/README.md).

## 9. Timing it honestly

A host clock around a launch measures only the time to queue it. Use **CUDA events**, which the GPU records on its own timeline:

```cpp
cudaEventRecord(e0);  kernel<<<...>>>(...);  cudaEventRecord(e1);
cudaEventSynchronize(e1);
float ms; cudaEventElapsedTime(&ms, e0, e1);
GB/s = bytes_moved / (ms * 1e6)
```

Two honesty rules. The **first** CUDA call creates the GPU context, which takes noticeable time, so warm up and time the later runs. And separate the three parts: H2D copy, kernel, D2H copy. [`examples/02_transfer_tax.cu`](examples/02_transfer_tax.cu) does this.

Compare with the roofline from P1.4. Vector add does 1 add for 12 bytes (two reads, one write of 4 bytes), an intensity of 1/12 ≈ 0.083 FLOP/byte. That is far left of any ridge, so its best time is **bytes ÷ memory bandwidth**. For n = 2²⁶: 12 × 67,108,864 = 805,306,368 bytes; at 320 GB/s (UNVERIFIED) that is about 2.52 ms. The same 805 MB over PCIe at 12 GB/s (example value) is about 67 ms. The copies cost about 27 times the kernel, which is 320 ÷ 12. The simulator in the lesson lets you move those numbers. It is a model, not a measurement.

## 10. Where to go next

- **Lane B L1** (weeks 6 to 10): many indexing problems, 1D, 2D and 3D, on LeetGPU with the local harness. Do P1.6 before L1.
- **P2 to P4** use GPUs through frameworks; Lane B L2 and L3 continue alongside.
- **P5** is the performance deep dive. It assumes this module and Lane B L1 to L3.

---

## Walkthrough

```bash
M=course/P1-inference-fundamentals/P1.6-first-cuda-program
cmake -S $M/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/p1.6 -j   # TODO(run-on: T2)
./build/p1.6/p1.6_01_device_info          # your GPU's compute capability, SM count, memory
./build/p1.6/p1.6_02_transfer_tax        # H2D, kernel and D2H timings and GB/s
```

## What you should see

- `01_device_info`: one block per GPU with name, compute capability (7.5 on a T4), SM count and memory. `TODO(run-on: T2)`: record yours.
- `02_transfer_tax`: two passes of `H2D | kernel | D2H` with GB/s. Expect the kernel row to be far faster than the copy rows, and the first pass to be slower than the second. `TODO(run-on: T2)`: paste your numbers next to the simulator's prediction.

## Exercises

```bash
ctest --test-dir build/p1.6 --output-on-failure        # starters fail until you implement them
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [**Hello, device**](exercises/01-hello-device/README.md): 2 blocks × 4 threads print their global index | easy (T2) | launch succeeds; indices recorded |
| 2 | [**Vector add, the full program**](exercises/02-vector-add-full/README.md) | medium (T2) | N in {1, 1000, 2²⁰, 2²⁰ + 3} vs a CPU loop |
| 3 | [**Error checks**](exercises/03-error-checks/README.md): a planted bad launch and a kernel fault | medium (T2) | the errors are reported, not swallowed |
| 4 | [**compute-sanitizer**](exercises/04-sanitizer/README.md): read the report for an out-of-bounds write | easy (T2) | none: paste and explain the first error |

To check your work against the references, configure a second build directory with `-DS2S_USE_SOLUTIONS=ON`. Every run here is `TODO(run-on: T2)`: no run result has been recorded in this repository yet.

## Common mistakes

- **Dereferencing a device pointer on the CPU.** It segfaults or reads garbage. Copy first.
- **Swapping `cudaMemcpy` arguments** or passing an element count instead of bytes. Destination first, bytes second-to-last.
- **No bounds check** after rounding the grid up. It often "works", which is the danger.
- **Reading the result before the kernel finishes**, with a launch you never synchronized (only a problem with async copies and streams; a plain `cudaMemcpy` waits).
- **Timing with a host clock** around the launch, or including the first call.
- **Comparing floats with `==`** in reductions (not in this module's add, which is exact; L3 uses tolerances).

## Go deeper

- NVIDIA blog, [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/).
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): *Programming Model* and *Programming Interface*.
- [Compute Sanitizer docs](https://docs.nvidia.com/compute-sanitizer/).
- No GPU yet? [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) in a notebook, [LeetGPU](https://leetgpu.com/) in the browser (link only), [Google Colab](https://colab.research.google.com/) for a notebook GPU.

All of these links were unreachable from the build environment (UNVERIFIED).

## Go down when…

You want the pinned-memory and streams story: [P5.1](../../P5-cuda-deep-dive/P5.1-execution-model/README.md). **Next:** [P2.1 First serve with vLLM](../../P2-serving-engines/P2.1-first-serve-with-vllm/README.md), and start [Lane B L1](../../../laneB-cuda/L1-launch-and-indexing/syllabus.md) alongside.
