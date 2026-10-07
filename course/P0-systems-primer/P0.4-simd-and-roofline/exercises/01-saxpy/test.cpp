#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

static void check_n(size_t n, size_t off) {
  std::vector<float> x(n + 8), y(n + 8), want(n + 8);
  for (size_t i = 0; i < n + 8; ++i) { x[i] = 0.5f * float(i % 13) - 2.0f; y[i] = float(i % 7); want[i] = y[i]; }
  for (size_t i = 0; i < n; ++i) want[off + i] = 1.5f * x[off + i] + want[off + i];
  saxpy(1.5f, x.data() + off, y.data() + off, n);
  CHECK_ALLCLOSE(y.data(), want.data(), n + 8, 1e-6, 1e-6);  // also checks nothing outside [off, off+n) changed
}

S2S_TEST(all_small_sizes_and_offsets) {
  for (size_t n = 0; n <= 67; ++n)
    for (size_t off = 0; off < 8; ++off) check_n(n, off);
}
S2S_TEST(large) { check_n(10000, 3); }

S2S_TEST_MAIN()
