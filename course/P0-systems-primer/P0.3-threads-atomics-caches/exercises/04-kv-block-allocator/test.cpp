#include <map>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(allocate_until_empty) {
  BlockAllocator a(3);
  CHECK_EQ(*a.allocate(), 0);  // block 0 first
  CHECK_EQ(*a.allocate(), 1);
  CHECK_EQ(*a.allocate(), 2);
  CHECK(!a.allocate().has_value());
  CHECK_EQ(a.num_free(), 0);
}

S2S_TEST(lifo_reuse) {
  BlockAllocator a(4);
  int x = *a.allocate(); int y = *a.allocate();
  a.free(x);
  CHECK_EQ(*a.allocate(), x);  // most recently freed comes back first
  (void)y;
}

S2S_TEST(allocate_many_all_or_nothing) {
  BlockAllocator a(5);
  CHECK_EQ(a.allocate_many(3).size(), size_t(3));
  CHECK(a.allocate_many(3).empty());  // only 2 left: take nothing
  CHECK_EQ(a.num_free(), 2);
}

S2S_TEST(share_and_free) {
  BlockAllocator a(2);
  int b = *a.allocate();
  a.share(b);
  CHECK_EQ(a.refcount(b), 2);
  a.free(b);
  CHECK_EQ(a.num_free(), 1);   // still referenced once
  a.free(b);
  CHECK_EQ(a.num_free(), 2);
}

S2S_TEST(double_free_throws) {
  BlockAllocator a(2);
  int b = *a.allocate();
  a.free(b);
  bool threw = false;
  try { a.free(b); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { a.share(7); } catch (const std::logic_error&) { threw = true; }
  CHECK(threw);
}

S2S_TEST(copy_on_write) {
  BlockAllocator a(4);
  int b = *a.allocate();
  CHECK_EQ(a.copy_on_write(b), b);  // sole owner: in place
  a.share(b);
  int c = a.copy_on_write(b);
  CHECK(c != b);
  CHECK_EQ(a.refcount(b), 1);
  CHECK_EQ(a.refcount(c), 1);
}

S2S_TEST(stress_invariants) {
  const int kBlocks = 256, kThreads = 8, kOps = 20000;
  BlockAllocator a(kBlocks);
  std::vector<std::vector<int>> held(kThreads);  // each thread's references (one entry per reference)
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&, t] {
      std::mt19937 rng(1234 + t);
      auto& mine = held[size_t(t)];
      for (int k = 0; k < kOps; ++k) {
        int op = int(rng() % 4);
        if (op == 0) {
          if (auto b = a.allocate()) mine.push_back(*b);
        } else if (op == 1 && !mine.empty()) {
          size_t i = rng() % mine.size();
          a.free(mine[i]);
          mine[i] = mine.back();
          mine.pop_back();
        } else if (op == 2) {
          // share a block that *we* still hold, so it cannot be freed under us
          if (!mine.empty()) { int b = mine[rng() % mine.size()]; a.share(b); mine.push_back(b); }
        } else if (op == 3 && !mine.empty()) {
          size_t i = rng() % mine.size();
          try { mine[i] = a.copy_on_write(mine[i]); } catch (const std::runtime_error&) { /* out of blocks: fine */ }
        }
      }
    });
  for (auto& th : ts) th.join();

  std::map<int, int> refs;
  for (auto& v : held) for (int b : v) ++refs[b];
  int live = 0;
  bool counts_ok = true;
  for (int b = 0; b < kBlocks; ++b) {
    int want = refs.count(b) ? refs[b] : 0;
    counts_ok &= a.refcount(b) == want;
    live += want > 0;
  }
  CHECK(counts_ok);
  CHECK_EQ(a.num_free() + live, kBlocks);
  for (auto& v : held) for (int b : v) a.free(b);
  CHECK_EQ(a.num_free(), kBlocks);
}

S2S_TEST_MAIN()
