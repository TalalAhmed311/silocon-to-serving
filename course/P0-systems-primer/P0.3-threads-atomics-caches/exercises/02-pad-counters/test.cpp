#include <cstdint>
#include <thread>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(one_line_per_slot) {
  CHECK_EQ(sizeof(Slot), size_t(64));
  CHECK_EQ(alignof(Slot), size_t(64));
  PerWorkerCounters<4> c;
  auto a = reinterpret_cast<uintptr_t>(&c.slots[0]), b = reinterpret_cast<uintptr_t>(&c.slots[1]);
  CHECK(a / 64 != b / 64);
}

S2S_TEST(exact_count) {
  PerWorkerCounters<4> c;
  std::vector<std::thread> ts;
  for (int w = 0; w < 4; ++w) ts.emplace_back([&, w] { for (int i = 0; i < 100000; ++i) c.inc(w); });
  for (auto& t : ts) t.join();
  CHECK_EQ(c.total(), uint64_t(400000));
}

S2S_TEST_MAIN()
