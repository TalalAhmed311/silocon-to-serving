#include <thread>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(exact_total_and_max) {
  TokenStats s;
  const int T = 8, N = 200000;
  std::vector<std::thread> ts;
  for (int t = 0; t < T; ++t)
    ts.emplace_back([&, t] { for (int i = 0; i < N; ++i) s.record(uint64_t((i * 7 + t) % 100)); });
  for (auto& th : ts) th.join();
  uint64_t want = 0;
  for (int t = 0; t < T; ++t) for (int i = 0; i < N; ++i) want += uint64_t((i * 7 + t) % 100);
  CHECK_EQ(s.get_total(), want);
  CHECK_EQ(s.get_max(), uint64_t(99));
}

S2S_TEST_MAIN()
