#include <thread>

#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(empty_and_full) {
  SpscQueue<int, 4> q;
  int x;
  CHECK(!q.pop(x));
  for (int i = 0; i < 4; ++i) CHECK(q.push(i));
  CHECK(!q.push(99));
  CHECK(q.pop(x));
  CHECK_EQ(x, 0);
}

S2S_TEST(ordered_no_loss) {
  SpscQueue<long, 1024> q;
  const long N = 2000000;
  bool ordered = true;
  long received = 0;
  std::thread consumer([&] {
    long expect = 0, v;
    while (expect < N) {
      if (q.pop(v)) { ordered &= v == expect; ++expect; ++received; }
    }
  });
  for (long i = 0; i < N; ++i) while (!q.push(i)) {}
  consumer.join();
  CHECK(ordered);
  CHECK_EQ(received, N);
}

S2S_TEST_MAIN()
