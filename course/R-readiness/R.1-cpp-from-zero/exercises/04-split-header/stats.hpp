// Exercise 4 starter (the DECLARATIONS). Callers include this file and learn the signatures.
// The DEFINITIONS (the code) live in stats.cpp, which is compiled once into a library.
#pragma once
#include <cstddef>

// Mean of a[0] .. a[n-1]. Returns 0 for n == 0.
double mean(const double* a, size_t n);

// Population variance: the mean of (a[i] - mean)^2. Returns 0 for n == 0.
double variance(const double* a, size_t n);

// Largest minus smallest element. Returns 0 for n == 0.
double range(const double* a, size_t n);

// A tiny function defined right here in the header. It must be marked `inline`, otherwise every
// .cpp that includes this file gets its own copy and the linker reports a duplicate definition.
// (Try it: delete the word `inline`, rebuild, and read the linker error. Then put it back.)
inline double square(double v) { return v * v; }
