// 04_thread_pool.cpp — parallel_for over a compute-bound loop; verifies the result and reports speedup.
// Run:      ./build/examples/04_thread_pool
// Expected: identical sums; speedup ≈ number of physical cores (less with SMT siblings).
// Hardware: T0.
#include <cmath>
#include <cstdio>
#include <vector>

#include <s2s/bench.hpp>
#include "thread_pool.hpp"

static double work(int64_t i) { return std::sin(double(i)) * std::cos(double(i) * 0.5); }  // compute-bound body

int main() {
  const int64_t n = 20'000'000;
  s2s::ThreadPool pool;
  double serial = 0, parallel = 0;
  auto ts = s2s::time_fn([&] { serial = 0; for (int64_t i = 0; i < n; ++i) serial += work(i); }, 1, 3);
  auto tp = s2s::time_fn([&] {
    std::vector<double> partial(pool.size() * 8, 0.0);  // spaced 8 doubles = 64 B apart: no false sharing
    pool.parallel_for(n, [&](int64_t b, int64_t e) {
      double s = 0;
      for (int64_t i = b; i < e; ++i) s += work(i);
      partial[size_t(b / ((n + pool.size() - 1) / pool.size())) * 8] = s;
    });
    parallel = 0;
    for (double v : partial) parallel += v;
  }, 1, 3);
  std::printf("workers=%u\n| run | sum | median ms |\n|---|---|---|\n", pool.size());
  std::printf("| serial | %.6f | %.1f |\n| parallel_for | %.6f | %.1f |\n", serial, ts.median_ms, parallel, tp.median_ms);
  std::printf("speedup %.2fx (sums differ only by floating-point summation order: |diff| = %.2e)\n",
              ts.median_ms / tp.median_ms, std::fabs(serial - parallel));
  return 0;
}
