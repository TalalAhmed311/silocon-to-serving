// false_sharing_bench.cpp — packed vs padded per-thread counters for 1..2*cores threads.
// Run:    ./build/bench/false_sharing_bench
// Output: | threads | packed ns/op | padded ns/op | speedup |  + results/false_sharing.json
// Hardware: T0.
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include <s2s/bench.hpp>

struct Packed { std::atomic<long> v{0}; };
struct alignas(64) Padded { std::atomic<long> v{0}; };
constexpr long kIters = 20'000'000;

template <class C>
double ns_per_op(int threads) {
  std::vector<C> c(size_t(threads));
  auto t = s2s::time_fn([&] {
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; ++i)
      ts.emplace_back([&, i] { for (long k = 0; k < kIters / threads; ++k) c[size_t(i)].v.fetch_add(1, std::memory_order_relaxed); });
    for (auto& th : ts) th.join();
  }, 1, 5);
  return t.median_ms * 1e6 / double(kIters);
}

int main() {
  const int maxT = int(std::max(2u, 2 * std::thread::hardware_concurrency()));
  std::printf("| threads | packed ns/op | padded ns/op | speedup |\n|---|---|---|---|\n");
  std::filesystem::create_directories("results");
  std::ofstream js("results/false_sharing.json");
  js << "{\"bench\": \"false_sharing\", \"unit\": \"ns/op\", \"rows\": [";
  bool first = true;
  for (int t = 1; t <= maxT; t *= 2) {
    double a = ns_per_op<Packed>(t), b = ns_per_op<Padded>(t);
    std::printf("| %d | %.2f | %.2f | %.1fx |\n", t, a, b, a / b);
    js << (first ? "" : ", ") << "{\"threads\": " << t << ", \"packed\": " << a << ", \"padded\": " << b << "}";
    first = false;
  }
  js << "]}\n";
  return 0;
}
