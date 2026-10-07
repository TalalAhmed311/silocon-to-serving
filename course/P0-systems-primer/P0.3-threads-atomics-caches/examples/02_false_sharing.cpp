// 02_false_sharing.cpp — per-thread counters packed into one cache line vs one line each.
// Run:      ./build/examples/02_false_sharing [threads]   (default 4)
// Expected: padded is several times faster than packed; the ratio grows with threads.
// Hardware: T0. Try `perf c2c record/report` on the packed run to see the contended line.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <s2s/bench.hpp>

constexpr int kLine = 64;  // std::hardware_destructive_interference_size where available
constexpr long kIters = 50'000'000;

struct Packed { std::atomic<long> v{0}; };                // 8 bytes: 8 of these share one line
struct alignas(kLine) Padded { std::atomic<long> v{0}; };  // 64 bytes: one line each
static_assert(sizeof(Padded) == kLine);

template <class Counter>
double run(int threads) {
  std::vector<Counter> c(threads);
  auto t = s2s::time_fn([&] {
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; ++i)
      ts.emplace_back([&, i] {
        // relaxed: we only need atomicity of each increment; no other data is published.
        for (long k = 0; k < kIters / threads; ++k) c[i].v.fetch_add(1, std::memory_order_relaxed);
      });
    for (auto& th : ts) th.join();
  }, 1, 5);
  return t.median_ms * 1e6 / double(kIters);  // ns per increment (total work is fixed)
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? std::atoi(argv[1]) : 4;
  double packed = run<Packed>(threads), padded = run<Padded>(threads);
  std::printf("| threads | packed ns/op | padded ns/op | speedup |\n|---|---|---|---|\n");
  std::printf("| %d | %.2f | %.2f | %.1fx |\n", threads, packed, padded, packed / padded);
  return 0;
}
