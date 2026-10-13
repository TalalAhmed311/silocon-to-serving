# R.2 — Memory: stack, heap and ownership

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Linux or macOS, x86 or ARM, no GPU |
| **Time** | ≈30 min reading + ≈4 h hands-on |
| **Prerequisites** | [R.1](../R.1-cpp-from-zero/README.md): pointers, references, constructors and destructors. `g++` or `clang++` ≥ 11 and CMake ≥ 3.20 |
| **You will build** | 4 tested exercises, all built with AddressSanitizer and UBSan: fix a leak, fix a dangling return, write an owning buffer, read a sanitizer report |

> **Already comfortable with the stack and heap, `new`/`delete`, leaks, `unique_ptr` and move?** Answer the [quiz](quiz.md) and the checkpoints in the [interactive lesson](../../../lessons/r-2.html). If you get them all right, skip to [R.3](../R.3-terminal-git-build/README.md).

## Learning objectives

By the end of this module you can:

1. Name the four regions of a process's memory and say who frees each.
2. Explain how stack frames are pushed and popped, and why returning a pointer to a local dangles.
3. Pair `new`/`delete`, `new[]`/`delete[]` and `malloc`/`free`.
4. Recognise a leak, a double free and a use-after-free from code and from an ASan report.
5. Compile with `-fsanitize=address,undefined -g` and read the first lines of a report.
6. Apply the rule "exactly one owner frees", using RAII and `std::unique_ptr`.
7. Say what copying and moving a `std::vector` each do to its heap buffer.

## Why this matters for inference engines

An engine owns gigabytes of weights and a KV cache, and it hands out blocks of them to requests. Each block must be freed exactly once, after its last user, or the server leaks, crashes or serves garbage. Ownership is the vocabulary that makes this reviewable.

---

## 1. A process's memory

Four regions: **code** (your functions), **globals** (global and `static` variables), the **stack** (locals, one frame per call) and the **heap** (blocks from `new` and `malloc`). Code and globals live as long as the program. The stack frees itself. The heap is freed by you. The addresses are virtual; [P0.2](../../P0-systems-primer/P0.2-virtual-memory-and-mmap/README.md) explains the translation.

## 2. The stack

A call pushes a frame (parameters, locals, return address). A return pops it. A local array of `float big[10000000]` is 40,000,000 bytes and overflows Linux's default 8 MiB stack (`ulimit -s` prints 8192 KiB).

## 3. Dangling pointers

```cpp
int* f() { int x = 42; return &x; }   // x dies at the closing brace
```

The caller gets an address no variable owns. The old value may linger, so the bug can hide, until another call reuses the bytes. `-Wall` warns about the simplest form. Fix it by returning the value.

## 4. The heap

```cpp
float* w = new float[4096];   // 16 KiB
delete[] w;
```

`new`↔`delete`, `new[]`↔`delete[]`, `malloc`↔`free`. Never mix. The pointer lives on the stack and dies with its frame. The block lives until it is freed.

## 5. Three classic bugs

| bug | example | effect |
|---|---|---|
| leak | `p = nullptr;` before `delete[] p` | memory grows until the process dies |
| double free | `delete[] p; delete[] p;` | allocator bookkeeping corrupted |
| use-after-free | `delete[] p; p[0] = 1;` | writes into memory someone else owns |

All three come from unclear ownership.

## 6. AddressSanitizer

```bash
g++ -std=c++20 -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer main.cpp -o app
./app
```

A report shaped like a real one (this is an example written for the course, not captured output):

```
==4242==ERROR: AddressSanitizer: heap-buffer-overflow
WRITE of size 4 at 0x502000000020 thread T0
    #0 make_ramp(unsigned long) impl.hpp:15
    #1 ramp_of_four test.cpp:5
0x502000000020 is located 0 bytes after
  16-byte region [0x502000000010,0x502000000020)
allocated by thread T0 here:
    #0 operator new[](unsigned long)
    #1 make_ramp(unsigned long) impl.hpp:14
```

Read it in order: kind of error, read or write and size, the first frame in your code, the distance from the block ("0 bytes after" is an off-by-one), who allocated it. ASan slows a program down (the Clang docs say about 2x). Use it for tests, not for benchmarks. On Linux it also reports leaks at exit (LeakSanitizer).

## 7. Ownership and RAII

Every heap block has exactly one owner, and the owner frees it. **RAII** ties the free to a destructor, which runs on every exit path (early return, exception, normal end):

```cpp
class FloatBuffer {
  float* p_;
 public:
  explicit FloatBuffer(size_t n) : p_(new float[n]) {}
  ~FloatBuffer() { delete[] p_; }
};
```

## 8. Smart pointers

`std::unique_ptr<T>` is the sole owner: one pointer wide, no counting, move-only. Create it with `std::make_unique`. `std::shared_ptr<T>` counts owners with an atomic reference count and frees at zero. It is bigger, copies cost an atomic update, and it blurs when anything is freed, so engines keep it off hot paths. A function that only uses a block takes `T&` or `T*`.

## 9. Copy versus move

A `std::vector` is a small header owning a heap buffer. `auto b = a;` allocates a second buffer and copies every element (1,000,000 floats = 4,000,000 bytes). `auto c = std::move(a);` hands the pointer to `c` and leaves `a` empty: no allocation, no element copied. This is the bridge to the move section of [P0.1](../../P0-systems-primer/P0.1-c-cpp-for-systems/README.md).

## 10. Alignment, in one paragraph

A type's alignment is the number its address must be a multiple of (`alignof(double) == 8`). `alignas(64)` starts a type on a cache-line boundary. P0.1 section 2 has the full treatment.

---

## Exercises

All four are built with `-fsanitize=address,undefined` in a Debug configuration. `-DS2S_USE_SOLUTIONS=ON` builds the reference solutions. The starters compile; the tests fail until you fix them.

```bash
cd course/R-readiness/R.2-memory-management
cmake -S exercises -B /tmp/r2-ex && cmake --build /tmp/r2-ex -j && ctest --test-dir /tmp/r2-ex --output-on-failure
cmake -S exercises -B /tmp/r2-sol -DS2S_USE_SOLUTIONS=ON && cmake --build /tmp/r2-sol -j && ctest --test-dir /tmp/r2-sol --output-on-failure
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Fix the leak](exercises/01-fix-the-leak/README.md) | easy | counts live heap blocks before and after each call |
| 2 | [Dangling return](exercises/02-dangling-return/README.md) | easy | scribbles on the stack, then reads the results |
| 3 | [An owning `FloatBuffer`](exercises/03-owning-buffer/README.md) | medium | static live-allocation counter, move semantics, vector growth |
| 4 | [Read a sanitizer report](exercises/04-sanitizer-read/README.md) | easy | runs under ASan; you also paste its first line into a comment |

## Common mistakes

- **`delete` on a `new[]` block** (or the reverse).
- **Using a pointer after `std::move`** or after the owner died.
- **Calling `delete` on a pointer that a `unique_ptr` owns**: a double free.
- **Returning `&local`**, or `.data()` of a local vector.
- **Benchmarking with ASan on.**

## Go further

- [learncpp.com](https://www.learncpp.com/): dynamic memory, smart pointers and move semantics.
- [CS50x](https://cs50.harvard.edu/x/), Week 4 "Memory".
- [Clang AddressSanitizer docs](https://clang.llvm.org/docs/AddressSanitizer.html): what it detects and its flags.
- [cppreference](https://en.cppreference.com/w/): `unique_ptr`, `shared_ptr`, `make_unique`, `std::move`.

All links are UNVERIFIED from the build environment; see `SOURCES.md`.

## Next

[R.3 The terminal, git and building code](../R.3-terminal-git-build/README.md). After the Readiness path, [P0.1](../../P0-systems-primer/P0.1-c-cpp-for-systems/README.md) builds strides, alignment and a move-only tensor on these ideas.
