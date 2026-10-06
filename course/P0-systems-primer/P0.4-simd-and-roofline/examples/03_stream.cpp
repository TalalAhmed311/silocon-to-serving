// 03_stream.cpp — STREAM-style bandwidth: copy, scale, add, triad over arrays ≫ L3, single thread and all threads.
// Run:      ./build/examples/03_stream [MiB per array]   (default 256; use ≥ 4× your L3)
// Expected: the four kernels within ~10–20% of each other; all-threads ≫ single-thread on most laptops
//           (one core cannot saturate the memory controllers). Triad (all threads) is the "B" of your roofline.
// Hardware: T0. Not the official STREAM benchmark (McCalpin) — same kernels, same byte-counting convention.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <s2s/aligned.hpp>
#include <s2s/bench.hpp>
#include "thread_pool.hpp"

int main(int argc, char** argv) {
  const size_t n = (size_t(argc > 1 ? std::atoi(argv[1]) : 256) << 20) / sizeof(double);
  s2s::aligned_buffer<double> a(n), b(n), c(n);
  s2s::ThreadPool pool;
  // First-touch in parallel so pages land near the threads that use them (NUMA, P0.3 §5).
  pool.parallel_for(int64_t(n), [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) { a[i] = 1; b[i] = 2; c[i] = 0; } });
  const double q = 3.0;

  struct K { const char* name; double bytes_per_elem; void (*fn)(double*, double*, double*, double, int64_t, int64_t); };
  K ks[] = {
    {"copy  c=a", 16, [](double* a, double*, double* c, double, int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) c[i] = a[i]; }},
    {"scale b=q*c", 16, [](double*, double* b, double* c, double q, int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) b[i] = q * c[i]; }},
    {"add   c=a+b", 24, [](double* a, double* b, double* c, double, int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) c[i] = a[i] + b[i]; }},
    {"triad a=b+q*c", 24, [](double* a, double* b, double* c, double q, int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) a[i] = b[i] + q * c[i]; }},
  };
  std::printf("arrays: 3 x %zu MiB\n\n| kernel | 1 thread GB/s | %u threads GB/s |\n|---|---|---|\n", n * 8 >> 20, pool.size());
  double triad_all = 0;
  for (auto& k : ks) {
    auto t1 = s2s::time_fn([&] { k.fn(a.data(), b.data(), c.data(), q, 0, int64_t(n)); }, 1, 5);
    auto tn = s2s::time_fn([&] { pool.parallel_for(int64_t(n), [&](int64_t lo, int64_t hi) { k.fn(a.data(), b.data(), c.data(), q, lo, hi); }); }, 1, 5);
    const double gb = k.bytes_per_elem * double(n) / 1e9;
    std::printf("| %s | %.1f | %.1f |\n", k.name, gb / (t1.median_ms / 1e3), gb / (tn.median_ms / 1e3));
    triad_all = gb / (tn.median_ms / 1e3);  // last kernel is triad
  }
  std::filesystem::create_directories("results");
  std::ofstream("results/stream.json") << "{\"bench\": \"stream\", \"unit\": \"GB/s\", \"triad_all_threads\": " << triad_all << "}\n";
  std::printf("\nB (triad, all threads) = %.1f GB/s -> results/stream.json\n", triad_all);
  return 0;
}
