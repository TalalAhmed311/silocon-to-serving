#pragma once
#include <cstddef>

double mean(const double* a, size_t n);
double variance(const double* a, size_t n);
double range(const double* a, size_t n);

inline double square(double v) { return v * v; }
