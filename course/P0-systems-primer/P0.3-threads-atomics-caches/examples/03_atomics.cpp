// 03_atomics.cpp — one shared counter, three ways: mutex, atomic, per-thread then sum.
// Run:      ./build/examples/03_atomics [threads]   (default 4)
// Expected: all three give the exact same total; time: per-thread < atomic < mutex.
// Hardware: T0.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <s2s/bench.hpp>

constexpr long kN = 20'000'000;

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 4;
  auto spawn = [&](auto body) {
    std::vector<std::thread> ts;
    for (int i = 0; i < T; ++i) ts.emplace_back(body, i);
    for (auto& t : ts) t.join();
  };

  long m_total = 0;
  std::mutex mu;
  auto tm = s2s::time_fn([&] {
    m_total = 0;
    spawn([&](int) { for (long k = 0; k < kN / T; ++k) { std::lock_guard g(mu); ++m_total; } });
  }, 1, 3);

  std::atomic<long> a_total{0};
  auto ta = s2s::time_fn([&] {
    a_total = 0;
    spawn([&](int) { for (long k = 0; k < kN / T; ++k) a_total.fetch_add(1, std::memory_order_relaxed); });
  }, 1, 3);

  long p_total = 0;
  auto tp = s2s::time_fn([&] {
    struct alignas(64) Slot { long v = 0; };   // padded, or we'd reintroduce false sharing (example 02)
    std::vector<Slot> slots(T);
    spawn([&](int i) { long local = 0; for (long k = 0; k < kN / T; ++k) ++local; slots[i].v = local; });
    p_total = 0;
    for (auto& s : slots) p_total += s.v;
  }, 1, 3);

  std::printf("| method | total | median ms |\n|---|---|---|\n");
  std::printf("| mutex | %ld | %.1f |\n| atomic fetch_add | %ld | %.1f |\n| per-thread + sum | %ld | %.2f |\n",
              m_total, tm.median_ms, a_total.load(), ta.median_ms, p_total, tp.median_ms);
  // Note: the compiler may turn the per-thread loop into `local = kN/T` outright; that is the point —
  // with no sharing, the work is no longer limited by coherence traffic at all.
  return 0;
}
