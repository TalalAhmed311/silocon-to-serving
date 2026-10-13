#pragma once
#include <cstddef>

// Declarations only. The definitions live in mathlib.cpp, which the build must compile and link.
int add(int a, int b);
long long fib(int n);                              // fib(0) = 0, fib(1) = 1
double mean(const double* v, std::size_t n);       // mean of n values; 0.0 when n == 0
