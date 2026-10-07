#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

// Reference written the slowest, most obvious way.
static std::vector<float> ref(const float* s, int64_t r, int64_t c, int64_t s0, int64_t s1) {
  std::vector<float> out;
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j) out.push_back(s[i * s0 + j * s1]);
  return out;
}

static std::vector<float> iota(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = float(i);
  return v;
}

static void run(const float* base, int64_t r, int64_t c, int64_t s0, int64_t s1) {
  std::vector<float> got(size_t(r * c), -1.0f);
  strided_copy(base, r, c, s0, s1, got.data());
  auto want = ref(base, r, c, s0, s1);
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(row_major) { auto b = iota(5 * 7); run(b.data(), 5, 7, 7, 1); }
S2S_TEST(column_major) { auto b = iota(5 * 7); run(b.data(), 5, 7, 1, 5); }
S2S_TEST(transposed) { auto b = iota(5 * 7); run(b.data(), 7, 5, 1, 7); }       // transpose of (5,7) row-major
S2S_TEST(every_other_column) { auto b = iota(6 * 10); run(b.data(), 6, 5, 10, 2); }
S2S_TEST(flipped_rows) {
  auto b = iota(4 * 3);
  run(b.data() + 3 * 3, 4, 3, -3, 1);  // start at the last row, step backwards
}
S2S_TEST(empty) { auto b = iota(1); run(b.data(), 0, 0, 0, 1); }

S2S_TEST_MAIN()
