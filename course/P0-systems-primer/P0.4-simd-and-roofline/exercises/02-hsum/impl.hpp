// Exercise 2 starter.
#pragma once
#include "simd.hpp"
#if defined(__AVX2__)
inline float hsum8(__m256 v) { (void)v; return 0.0f; /* TODO */ }
#elif defined(__ARM_NEON)
inline float hsum4(float32x4_t v) { (void)v; return 0.0f; /* TODO */ }
#endif
