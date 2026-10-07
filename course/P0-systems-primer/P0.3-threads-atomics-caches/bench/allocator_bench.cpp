// allocator_bench.cpp — throughput of your D2 BlockAllocator (alloc+free pairs) vs thread count.
// Run:    ./build/bench/allocator_bench        (-DS2S_USE_SOLUTIONS=ON at configure time to bench the reference)
// Output: | threads | ops/s | ns/op |  — a single mutex stops scaling quickly; that is the lesson (and why vLLM's
//         block pool is driven by one scheduler thread rather than many).
// Hardware: T0.
#include <cstdio>
#include <thread>
#include <vector>

#include <s2s/bench.hpp>
#include "impl.hpp"

int main() {
  const long kOps = 2'000'000;
  std::printf("| threads | alloc+free pairs/s | ns per pair |\n|---|---|---|\n");
  for (int t = 1; t <= int(std::max(2u, std::thread::hardware_concurrency())); t *= 2) {
    BlockAllocator a(4096);
    auto tm = s2s::time_fn([&] {
      std::vector<std::thread> ts;
      for (int i = 0; i < t; ++i)
        ts.emplace_back([&] { for (long k = 0; k < kOps / t; ++k) if (auto b = a.allocate()) a.free(*b); });
      for (auto& th : ts) th.join();
    }, 1, 3);
    std::printf("| %d | %.3g | %.1f |\n", t, kOps / (tm.median_ms / 1e3), tm.median_ms * 1e6 / kOps);
  }
  return 0;
}
