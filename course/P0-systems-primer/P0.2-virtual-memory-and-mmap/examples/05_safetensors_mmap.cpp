// 05_safetensors_mmap.cpp — list tensors in a .safetensors file and checksum them, without copying any weights.
// Run:      ./build/examples/05_safetensors_mmap /tmp/tiny.safetensors
// Expected: the same names, dtypes, shapes and sums that make_tiny_safetensors.py printed.
// Hardware: T0.
#include <cstdio>
#include <cstring>

#include "safetensors_view.hpp"

// IEEE half -> float, bit by bit (no compiler extension needed). Handles normals, subnormals, inf/nan.
static float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {  // subnormal: renormalize
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; --exp; }
      bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
  else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s file.safetensors\n", argv[0]); return 2; }
  st::File f(argv[1]);
  for (auto& [name, t] : f.tensors()) {
    int64_t n = 1;
    for (auto d : t.shape) n *= d;
    const uint8_t* p = f.bytes(t);
    double sum = 0;
    // memcpy per element: the safe way to read a possibly unaligned typed value from a byte pointer.
    for (int64_t i = 0; i < n; ++i) {
      if (t.dtype == "F32") { float v; std::memcpy(&v, p + 4 * i, 4); sum += v; }
      else if (t.dtype == "F16") { uint16_t v; std::memcpy(&v, p + 2 * i, 2); sum += half_to_float(v); }
      else if (t.dtype == "I32") { int32_t v; std::memcpy(&v, p + 4 * i, 4); sum += v; }
    }
    std::printf("%-24s %-8s [", name.c_str(), t.dtype.c_str());
    for (size_t k = 0; k < t.shape.size(); ++k) std::printf("%s%lld", k ? ", " : "", (long long)t.shape[k]);
    std::printf("]  offset=%llu  sum=%.6f\n", (unsigned long long)t.begin, sum);
  }
  return 0;
}
