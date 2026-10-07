#include <atomic>
#include <stdexcept>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

static int64_t par_sum(ThreadPool& p, int64_t n) {
  std::vector<int64_t> out(size_t(n), 0);
  p.parallel_for(n, [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) out[size_t(i)] = i; });
  int64_t s = 0;
  for (auto v : out) s += v;
  return s;
}

S2S_TEST(sums_match_serial) {
  ThreadPool p(4);
  for (int64_t n : {0, 1, 3, 1000, 1000000}) CHECK_EQ(par_sum(p, n), n * (n - 1) / 2);
}

S2S_TEST(every_index_exactly_once) {
  ThreadPool p(3);
  std::vector<std::atomic<int>> hits(997);
  p.parallel_for(997, [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; ++i) hits[size_t(i)]++; });
  bool ok = true;
  for (auto& h : hits) ok &= h.load() == 1;
  CHECK(ok);
}

S2S_TEST(many_back_to_back_calls) {
  ThreadPool p(4);
  std::atomic<int64_t> total{0};
  for (int k = 0; k < 1000; ++k) p.parallel_for(10, [&](int64_t b, int64_t e) { total += e - b; });
  CHECK_EQ(total.load(), int64_t(10000));
}

S2S_TEST(exception_propagates_and_pool_survives) {
  ThreadPool p(4);
  bool threw = false;
  try {
    p.parallel_for(100, [&](int64_t b, int64_t) { if (b == 0) throw std::runtime_error("boom"); });
  } catch (const std::runtime_error&) { threw = true; }
  CHECK(threw);
  CHECK_EQ(par_sum(p, 100), int64_t(4950));
}

S2S_TEST_MAIN()
