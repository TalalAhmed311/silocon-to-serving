# R.1 — C and C++ from zero

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Linux or macOS, x86 or ARM, no GPU |
| **Time** | ≈30 min reading + ≈4 h hands-on |
| **Prerequisites** | You can write a loop and a function in some language (Python or JavaScript is fine). You have `g++` or `clang++` ≥ 11 and CMake ≥ 3.20 |
| **You will build** | 4 tested exercises: swap, pointer-walk sum and max, a `Vec2` class, a two-file module |

> **Already comfortable with pointers, references, `const`, headers and destructors?** Answer the [quiz](quiz.md) and the checkpoints in the [interactive lesson](../../../lessons/r-1.html). If you get them all right, skip to [R.2](../R.2-memory-management/README.md).

## Learning objectives

By the end of this module you can:

1. Explain what the compiler and the linker each do, and tell a compile error from a link error.
2. State the sizes of `char`, `int`, `float`, `double` and a pointer, and compute `sizeof` of an array.
3. Use `&x`, `*p` and `p + i`, and say how many bytes `p + i` moves for any type.
4. Pass an argument by value, by reference and by pointer, and choose `const T&` for big read-only inputs.
5. Split code into a header and a source file, and avoid the one-definition error.
6. Write a small class with a constructor, a `const` method and an operator, and say when its destructor runs.
7. Use `std::vector`, `std::array`, `std::string` and range-for.
8. Explain undefined behaviour, and why the rest of the course builds with sanitizers.

## Why this matters for inference engineering

P0.1 starts with "memory is a line of bytes", then moves straight to alignment, strides, RAII and move semantics. That is a steep start if pointers and headers are new. Every engine, kernel and tool in the course is written in C or C++. This module gives you the language floor so P0.1 can focus on memory itself.

---

## 1. Source, compiler, object file, linker

Python runs your source line by line. C++ is translated ahead of time. The **compiler** turns each `.cpp` file into an **object file** (`.o`) of machine code. The **linker** joins object files and libraries into one executable.

```bash
g++ -c main.cpp    -o main.o
g++ -c mathlib.cpp -o mathlib.o
g++ main.o mathlib.o -o app
./app
```

An object file is not runnable. `main.o` records "I call `square`; address unknown". `mathlib.o` records "I define `square`". The linker matches them. Two kinds of failure follow: a *compile error* (one file is wrong) and a *link error* such as `undefined reference to 'square'` (a function was promised, never defined).

## 2. Types have sizes

| type | bytes (64-bit Linux, macOS, Windows) |
|---|---|
| `char` | 1 |
| `int`, `float` | 4 |
| `double` | 8 |
| any pointer | 8 |

`sizeof(T)` asks the compiler. An array is its elements back to back: `int a[10]` is 40 bytes. These sizes turn into memory bills: 8 billion `float` weights are 32 GB.

## 3. Variables have addresses; pointers store them

```cpp
int x = 7;
int* p = &x;   // p holds the address of x
*p = 9;        // follow the pointer: x is now 9
```

The `*` in `int* p` is part of the type. The `*` in `*p = 9` follows the pointer. `nullptr` means "points at nothing"; never follow it.

## 4. Arrays and pointer arithmetic

If `int a[5]` starts at `0x1000`, element `i` is at `0x1000 + i × 4`. Adding to a pointer counts in elements:

```
p + i  =  address in p  +  i × sizeof(element)
a[i]   is defined as   *(a + i)
```

C++ does not check bounds. The [lesson's simulator](../../../lessons/r-1.html) lets you try other types, base addresses and indexes.

## 5. References, pointers and `const`

```cpp
void f(int  x)  { x += 1; }    // copy
void g(int& x)  { x += 1; }    // alias for the caller's variable
void h(int* x)  { *x += 1; }   // address of the caller's variable
```

A reference cannot be null and cannot be re-pointed. Use a pointer when "no object" is valid or when you walk an array. For big read-only inputs, write `const T&`: no copy, and the compiler stops writes. Small cheap types (`int`, `float`, pointers) go by value.

## 6. Headers, source files and the one-definition rule

A **declaration** (`double mean(const double*, size_t);`) lives in a `.hpp`. The **definition** (the code) lives in exactly one `.cpp`. `#include "stats.hpp"` pastes the header's text into the file. A function defined in a header must be `inline`, or every includer defines it again and the linker reports `multiple definition`. Start headers with `#pragma once`.

## 7. Classes, constructors, destructors

```cpp
struct Vec2 {
  float x, y;
  Vec2(float a, float b) : x(a), y(b) {}
  float length() const { return std::sqrt(x * x + y * y); }
};
```

The constructor runs when the object is created. A `const` after the parameter list promises not to modify the object. The destructor `~Name()` runs automatically at the closing brace, in reverse order of construction. [R.2](../R.2-memory-management/README.md) builds on that.

## 8. The standard library you will meet first

`std::vector<T>` is a growable array (a 24-byte header plus a heap block). `std::array<T, N>` keeps `N` elements inline. `std::string` owns characters. `for (const auto& x : v)` loops without an index and without copies.

## 9. Undefined behaviour

Reading `a[4]` from `int a[4]`, or reading a variable you never set, compiles. The language promises nothing about what happens next. Use `-Wall -Wextra` to catch simple cases. Use `-fsanitize=address,undefined -g` in tests to catch the rest. R.2 shows how to read the reports.

---

## Exercises

All four build as one CMake project. Run from this folder. `-DS2S_USE_SOLUTIONS=ON` builds the reference solutions instead of your starters, so you can confirm your toolchain first. The starters compile; their tests fail until you write the code.

```bash
cd course/R-readiness/R.1-cpp-from-zero
cmake -S exercises -B /tmp/r1-ex && cmake --build /tmp/r1-ex -j && ctest --test-dir /tmp/r1-ex --output-on-failure
cmake -S exercises -B /tmp/r1-sol -DS2S_USE_SOLUTIONS=ON && cmake --build /tmp/r1-sol -j && ctest --test-dir /tmp/r1-sol --output-on-failure
```

| # | Exercise | Difficulty | Test |
|---|---|---|---|
| 1 | [Swap by pointer and by reference](exercises/01-swap/README.md) | easy | values exchanged, self-swap |
| 2 | [Sum and max with pointer arithmetic](exercises/02-array-sum/README.md) | easy | small arrays, negatives, `n == 0`, sub-ranges |
| 3 | [A `Vec2` class](exercises/03-vec2-class/README.md) | easy | constructors, `const` call, `operator+`, `sizeof == 8` |
| 4 | [Split a module into `.hpp` and `.cpp`](exercises/04-split-header/README.md) | medium | links against a library target |

## Common mistakes

- **Treating `p + 1` as "add one byte".** It adds `sizeof(*p)` bytes.
- **Forgetting `const`** on a method, then wondering why it will not compile on a `const` object.
- **Defining a function in a header without `inline`.**
- **Using `*p` when `p` may be `nullptr`.**
- **Mixing signed and unsigned**: `v.size()` is unsigned, so `v.size() - 1` on an empty vector is a huge number.

## Go further

- [learncpp.com](https://www.learncpp.com/): the chapters on pointers, references, `const`, dynamic allocation and classes.
- [cppreference](https://en.cppreference.com/w/): exact reference for every function and container.
- [CS50x](https://cs50.harvard.edu/x/), Week 4 "Memory": a video route to pointers and memory.

All three links are UNVERIFIED from the build environment; see `SOURCES.md`.

## Next

[R.2 Memory: stack, heap and ownership](../R.2-memory-management/README.md).
