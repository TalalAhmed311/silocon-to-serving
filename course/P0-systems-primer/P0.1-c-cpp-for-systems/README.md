# P0.1 — C/C++ for systems

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Linux or macOS, x86 or ARM, no GPU |
| **Time** | ≈30 min reading + ≈6 h hands-on |
| **Prerequisites** | You can write a loop and a function in some language; you have `g++`/`clang++` ≥ 11 and CMake ≥ 3.20 |
| **You will build** | 5 examples, 5 tested exercises, a transpose benchmark |

## Learning objectives

By the end of this module you can:

1. Predict `sizeof` and every field offset of a struct from alignment rules, and say why it matters for caches.
2. Walk a flat buffer as row-major, column-major or transposed **using strides alone**. This is exactly how PyTorch tensors work.
3. Write an RAII owner with move semantics, and *prove* when a copy happens by counting.
4. Write a small function template, a `constexpr` helper and a C++20 concept, and use `std::span` and `std::unique_ptr` correctly.
5. Build a multi-target project with modern CMake: targets, per-target flags, sanitizers and CTest.

## Why this matters for inference engineering

An LLM is about 99% big flat arrays of numbers. Every layer of the stack is a different way of slicing those arrays:

- vLLM's KV cache
- a safetensors file
- a CUDA kernel's shared-memory tile
- the weights your CPU engine `mmap`s in P0.5

If you can say exactly where element `(i, j)` lives in memory and how many bytes a struct really takes, half of performance engineering is already done.

---

## 1. Memory is a line of bytes

The machine sees memory as one long line of bytes, each with an address. Every variable, struct and array is a run of bytes somewhere on that line. A **pointer** is just one of those addresses together with a type. The type tells the compiler how many bytes to step for `p + 1`.

```c
float a[4] = {1, 2, 3, 4};
float *p = a;     // p holds the address of a[0]
p + 1;            // address + sizeof(float) = +4 bytes, i.e. &a[1]
*(p + 2) == a[2]; // pointer arithmetic and indexing are the same thing
```

## 2. Alignment and padding

A type's **alignment** is the address multiple it must start at. On mainstream 64-bit ABIs, `double` and pointers are 8-aligned and `float` and `int` are 4-aligned. The compiler inserts **padding** so every field is aligned, and pads the struct's total size up to a multiple of its largest field alignment, so that arrays of the struct stay aligned.

```c
struct Bad  { char tag; double value; char flag; };  // 1 + 7 pad + 8 + 1 + 7 pad = 24 bytes
struct Good { double value; char tag; char flag; };  // 8 + 1 + 1 + 6 pad          = 16 bytes
```

The same data costs 50% more memory in the bad order. For a metadata array touched every decode step, such as a block table entry or a request record, that is 50% more cache lines pulled from DRAM. `examples/01_layout.c` prints `sizeof` and `offsetof` so you can check every prediction.

Rules of thumb:

- order fields from largest alignment to smallest
- use `alignas(64)` when you want a field to own a whole cache line (P0.3 shows why)
- never assume a layout; check it with `offsetof` and `static_assert`

## 3. Shapes and strides: one buffer, many views

A 2-D matrix of shape `(rows, cols)` stored **row-major** puts element `(i, j)` at offset `i*cols + j`. Stored **column-major**, it is at `i + j*rows`. Both are special cases of one rule:

```
offset(i, j) = i * stride0 + j * stride1
```

| View | stride0 | stride1 |
|---|---|---|
| row-major `(R, C)` | C | 1 |
| column-major `(R, C)` | 1 | R |
| transpose of a row-major `(R, C)`, seen as `(C, R)` | 1 | C |

Transposing is free: swap the strides and no bytes move. The cost shows up later. When code walks the transposed view in the "natural" order, consecutive accesses jump `C * sizeof(float)` bytes. Each one lands on a different 64-byte cache line, so most of every line you fetch is wasted. PyTorch's `.contiguous()` exists to pay that cost once, by copying into a fresh buffer. `examples/02_strides.c` builds all three views of one buffer, and exercise 2 asks you to write `strided_copy`, which is `.contiguous()` for 2-D.

> **Predict first.** You transpose a 4096×4096 `float` matrix with the naive double loop: read row-major, write column-major. The read stream is contiguous. Each write lands 16 KB (4096 × 4 bytes) after the previous one, so every write touches a new cache line, and only 4 of that line's 64 bytes are useful. **What fraction of a plain copy's GB/s do you expect?** Write your guess down. Then run `bench/transpose_bench` and compare it with the "copy" row, which the same program measures.

## 4. RAII: the destructor is your `free`

**RAII** (Resource Acquisition Is Initialization) means a resource is owned by an object whose destructor releases it. When the object goes out of scope, by normal return or by exception, the resource is released. No leaks, and no `goto cleanup`.

```cpp
class Buffer {
  float* p_;
 public:
  explicit Buffer(size_t n) : p_(new float[n]) {}
  ~Buffer() { delete[] p_; }
};
```

This class has a bug: copying a `Buffer` copies the pointer, so two destructors free the same memory, a **double free**. You have three honest options:

1. **Delete copy.** `Buffer(const Buffer&) = delete;`
2. **Deep copy.** Allocate and `memcpy` in the copy constructor. This is correct but can be expensive by accident.
3. **Move.** Transfer ownership: the new object takes the pointer and the old one is set to `nullptr`.

Systems code usually picks **1 + 3: move-only**. That is what `std::unique_ptr` is, and what `course/common/include/s2s/aligned.hpp` implements. `examples/03_raii_buffer.cpp` counts constructions, copies and moves, so you can *see* which one each line triggers.

### Copy vs move, measured

`examples/04_move_vs_copy.cpp` runs one pipeline that passes a 64 MB buffer through 4 stages twice: once by value with copies, once with `std::move`. The copy version moves 4 × 64 MB through memory. The move version moves four pointers. Run it and look at the ratio. This is why an inference engine passes tensors by handle and never by value.

## 5. Templates, `constexpr`, concepts, `std::span`

- A **function template** is compiled once per type you use it with: `transpose<float>` and `transpose<int8_t>` are separate, fully optimized functions. This is how a kernel library supports fp32/fp16/int8 without a runtime `switch`.
- **`constexpr`** functions run at compile time when their inputs are constants. Use them for tile sizes and shape arithmetic, so mistakes become compile errors.
- A **concept** names the requirements on a template parameter. For example, `std::floating_point T` gives a readable error instead of a page of template noise.
- **`std::span<T>`** is a non-owning (pointer, length) view. Pass spans into functions, keep owners (`unique_ptr`, `aligned_buffer`) outside them.

`examples/05_templates.cpp` shows a `MatrixView<T>` (pointer + shape + strides) and a `constexpr` tile chooser.

## 6. Modern CMake in five lines

```cmake
add_library(tensor INTERFACE)                       # a target, not a global variable soup
target_include_directories(tensor INTERFACE include)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE tensor)           # usage requirements flow through targets
target_compile_options(app PRIVATE -O3 -march=native)
```

`examples/CMakeLists.txt` adds an `S2S_SANITIZE` option that turns on AddressSanitizer and UndefinedBehaviorSanitizer for every target. You will want it in exercise 5.

---

## Walkthrough: run the examples

```bash
cd course/P0-systems-primer/P0.1-c-cpp-for-systems
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release
cmake --build build/examples -j
./build/examples/01_layout
./build/examples/02_strides
./build/examples/03_raii_buffer
./build/examples/04_move_vs_copy
./build/examples/05_templates
```

## What you should see

`01_layout` prints the sizes below on x86-64 Linux, macOS arm64 and Windows x64. They are fixed by the ABI, so these are exact values, not measurements:

```
Bad   sizeof=24 offsets: tag=0 value=8 flag=16
Good  sizeof=16 offsets: value=0 tag=8 flag=9
Line  sizeof=64 (alignas(64))
```

`03_raii_buffer` prints its counters. The exact numbers are asserted in the code, so the run aborts if your compiler disagrees:

```
make a          -> ctor=1 copy=0 move=0
b = std::move(a)-> ctor=1 copy=0 move=1
v.push_back(..) -> ...
```

`04_move_vs_copy` and the transpose bench print timings that depend on your machine. **TODO(run): paste your table here.** The expected shape:

- the copy pipeline is much slower than the move pipeline, roughly in proportion to bytes copied
- the naive transpose is a small fraction of copy bandwidth
- the blocked transpose recovers most of the gap

## Exercises

All exercises build with one CMake project. `-DS2S_USE_SOLUTIONS=ON` builds the reference solutions instead of your starters, so you can confirm the tests pass.

```bash
cmake -S exercises -B build/ex && cmake --build build/ex -j && ctest --test-dir build/ex --output-on-failure
cmake -S exercises -B build/ex-sol -DS2S_USE_SOLUTIONS=ON && cmake --build build/ex-sol -j && ctest --test-dir build/ex-sol
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Shrink a struct](exercises/01-shrink-struct/README.md) | easy | `static_assert` + `sizeof` checks |
| 2 | [`strided_copy` = `.contiguous()`](exercises/02-strided-copy/README.md) | easy | compared against a naive loop for 4 view types |
| 3 | [The `Tensor` RAII class](exercises/03-tensor-raii/README.md) | medium | move counts, ownership, 64-byte alignment |
| 4 | [Generic blocked `transpose<T, Block>`](exercises/04-blocked-transpose/README.md) | medium | `float` and `int8_t`, odd shapes |
| 5 | [Fix three planted memory bugs](exercises/05-sanitizer-hunt/README.md) | hard | runs under ASan + UBSan |

## Benchmark

```bash
cmake -S bench -B build/bench -DCMAKE_BUILD_TYPE=Release && cmake --build build/bench -j
./build/bench/transpose_bench            # prints the table, writes results/transpose.json
```

Columns: `variant | problem size | median ms | p90 ms | GB/s | % of peak`. "Peak" is the `copy` row, measured in the same run.

## Common mistakes

- **Assuming `sizeof` is the sum of the fields.** It almost never is. Use `offsetof`.
- **`std::vector<float>` for SIMD data.** Its default alignment is only `alignof(float)`. Use an aligned allocator (`s2s::aligned_buffer`) when you need 32/64-byte alignment.
- **Rule of three/five violations.** If you write a destructor, decide about copy and move explicitly.
- **Passing big objects by value "because it's simpler".** Pass `const T&`, `std::span`, or move.
- **Benchmarking a Debug build.** Always `-DCMAKE_BUILD_TYPE=Release`, and check that the compiler didn't delete your loop (`s2s::do_not_optimize`).

## Go deeper

- CS:APP, ch. 6 "The Memory Hierarchy" (§6.2 locality, §6.5–6.6 cache-friendly code). ch. 3.9 covers struct alignment.
- learncpp.com: chapters on move semantics and smart pointers (22.x), and templates (11.x, 26.x).
- *C++ Concurrency in Action*, ch. 1, for the toolchain setup used in P0.3.
- llama2.c `run.c` (`karpathy/llama2.c@350e04fe`): its `Config` and `TransformerWeights` structs and `memory_map_weights()` slice one big buffer into named tensors with pointer arithmetic. Read it now as a worked example of §1–3. You will build your own version in P0.5.

## Go down when…

This is the bottom of the CPU path. **Next:** [P0.2 Virtual memory and mmap](../P0.2-virtual-memory-and-mmap/README.md).

## Animation

[`animations/p0-memory-layout.html`](../../../animations/p0-memory-layout.html) shows struct padding byte by byte, and the same buffer under row-major, column-major and transposed strides. Step through the access order and watch which cache lines each pattern touches.
