// 02_fma_peak.cpp — measure peak fp32 GFLOP/s: sweep the number of independent FMA accumulators, then all cores.
// Run:      ./build/examples/02_fma_peak            (writes results/peak.json for the roofline plot)
// Expected: GFLOP/s grows with accumulators and plateaus near (FMA latency × FMA units) accumulators —
//           typically 8–10 on x86. The plateau is your single-core peak; the last row multiplies by cores.
// Hardware: T0.
#include <cstdio>
#include <filesystem>
#include <fstream>

#include <s2s/bench.hpp>
#include "simd.hpp"
#include "thread_pool.hpp"

constexpr long kIters = 20'000'000;

// K independent dependency chains. The compiler keeps each accumulator in its own register.
template <int K>
static float fma_chains(float x) {
  simd::vf acc[K], m = simd::set1(0.999999f), a = simd::set1(x * 1e-7f);
  for (int k = 0; k < K; ++k) acc[k] = simd::set1(float(k));
  for (long i = 0; i < kIters; ++i)
    for (int k = 0; k < K; ++k) acc[k] = simd::fma(acc[k], m, a);  // acc = acc*m + a: data-dependent per chain
  simd::vf s = simd::zero();
  for (int k = 0; k < K; ++k) s = simd::add(s, acc[k]);
  return simd::hsum(s);
}

template <int K>
static double gflops_one_core() {
  auto t = s2s::time_fn([] { s2s::do_not_optimize(fma_chains<K>(1.0f)); }, 1, 3);
  return 2.0 * simd::W * K * double(kIters) / (t.median_ms * 1e6);  // 2 FLOP per lane per FMA
}

int main() {
  std::printf("SIMD: %s\n\n| accumulators | GFLOP/s (1 core) |\n|---|---|\n", simd::name());
  double best = 0;
  auto row = [&](int k, double g) { std::printf("| %d | %.1f |\n", k, g); best = g > best ? g : best; };
  row(1, gflops_one_core<1>());
  row(2, gflops_one_core<2>());
  row(4, gflops_one_core<4>());
  row(6, gflops_one_core<6>());
  row(8, gflops_one_core<8>());
  row(10, gflops_one_core<10>());
  row(12, gflops_one_core<12>());

  s2s::ThreadPool pool;
  auto t = s2s::time_fn([&] {
    pool.parallel_for(pool.size(), [](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) s2s::do_not_optimize(fma_chains<10>(float(i))); });
  }, 1, 3);
  const double all = 2.0 * simd::W * 10 * double(kIters) * pool.size() / (t.median_ms * 1e6);
  std::printf("| 10 x %u threads | %.1f |\n\nmeasured peak: %.1f GFLOP/s per core, %.1f GFLOP/s all threads\n", pool.size(), all, best, all);
  std::filesystem::create_directories("results");
  std::ofstream("results/peak.json") << "{\"bench\": \"peak\", \"unit\": \"GFLOP/s\", \"per_core\": " << best
                                      << ", \"all_cores\": " << all << ", \"threads\": " << pool.size() << "}\n";
  return 0;
}
