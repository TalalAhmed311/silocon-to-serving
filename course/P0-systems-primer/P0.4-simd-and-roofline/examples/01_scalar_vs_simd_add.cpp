// 01_scalar_vs_simd_add.cpp — c = a + b, scalar vs SIMD, at in-cache and out-of-cache sizes; and a cache-line
// split penalty measurement.
// Run:      ./build/examples/01_scalar_vs_simd_add
// Expected: in L1/L2, SIMD is several times faster than scalar (up to ~W×). At 64 MiB both are limited by DRAM
//           bandwidth and run at about the same speed — the kernel is memory-bound (intensity 1/12 FLOP/B).
//           Split loads cost little or nothing extra on recent cores; up to ~2x on older ones.
// Hardware: T0. Compiled with -fno-tree-vectorize so "scalar" really is scalar.
#include <cstdio>
#include <vector>

#include <s2s/aligned.hpp>
#include <s2s/bench.hpp>
#include "simd.hpp"

static void add_scalar(const float* a, const float* b, float* c, size_t n) {
  for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

static void add_simd(const float* a, const float* b, float* c, size_t n) {
  size_t i = 0;
  for (; i + simd::W <= n; i += simd::W) simd::store(c + i, simd::add(simd::load(a + i), simd::load(b + i)));
  for (; i < n; ++i) c[i] = a[i] + b[i];  // tail
}

int main() {
  std::printf("SIMD: %s\n\n| n (floats) | footprint | scalar GB/s | SIMD GB/s | speedup |\n|---|---|---|---|---|\n", simd::name());
  for (size_t n : {size_t(1) << 10, size_t(1) << 14, size_t(1) << 18, size_t(1) << 24}) {
    s2s::aligned_buffer<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) { a[i] = float(i); b[i] = 1.0f; c[i] = 0.0f; }
    const int reps = int(std::max<size_t>(1, (size_t(1) << 26) / n));  // ~same total work per size
    auto ts = s2s::time_fn([&] { for (int r = 0; r < reps; ++r) add_scalar(a.data(), b.data(), c.data(), n); s2s::do_not_optimize(c[0]); });
    auto tv = s2s::time_fn([&] { for (int r = 0; r < reps; ++r) add_simd(a.data(), b.data(), c.data(), n); s2s::do_not_optimize(c[0]); });
    const double bytes = 12.0 * double(n) * reps;  // read a, read b, write c
    std::printf("| %zu | %.0f KiB | %.1f | %.1f | %.1fx |\n", n, 12.0 * n / 1024, bytes / ts.median_ms / 1e6,
                bytes / tv.median_ms / 1e6, ts.median_ms / tv.median_ms);
  }

  // Cache-line split: a 32-byte load starting 48 bytes into a 64-byte line touches two lines.
  const size_t n = 1 << 12;  // in L1, so we measure the load unit, not DRAM
  s2s::aligned_buffer<float> buf(n + 64);
  for (size_t i = 0; i < buf.size(); ++i) buf[i] = 1.0f;
  auto sum_at = [&](size_t offset_floats) {
    simd::vf acc = simd::zero();
    const float* p = buf.data() + offset_floats;
    for (int r = 0; r < 2000; ++r)
      for (size_t i = 0; i + 16 <= n; i += 16) acc = simd::add(acc, simd::load(p + i));  // one load per line
    return simd::hsum(acc);
  };
  auto t_al = s2s::time_fn([&] { s2s::do_not_optimize(sum_at(0)); });
  auto t_split = s2s::time_fn([&] { s2s::do_not_optimize(sum_at(12)); });  // 12 floats = 48 bytes into the line
  std::printf("\n| load placement | median ms |\n|---|---|\n| within one line | %.2f |\n| splits two lines | %.2f |\n",
              t_al.median_ms, t_split.median_ms);
  return 0;
}
