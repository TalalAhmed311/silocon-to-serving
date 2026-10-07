#include <cstdint>
#include <vector>

#include <s2s/check.hpp>
#include "impl.hpp"

template <class T>
static void check_shape(int64_t r, int64_t c) {
  std::vector<T> src(size_t(r * c)), got(size_t(r * c)), want(size_t(r * c));
  for (int64_t k = 0; k < r * c; ++k) src[size_t(k)] = T(k % 127);  // fits int8
  for (int64_t i = 0; i < r; ++i)
    for (int64_t j = 0; j < c; ++j) want[size_t(j * r + i)] = src[size_t(i * c + j)];
  transpose<T>(src.data(), got.data(), r, c);
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
  // A different block size must give the same answer.
  std::vector<T> got8(size_t(r * c));
  transpose<T, 8>(src.data(), got8.data(), r, c);
  CHECK_ALLCLOSE(got8.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(float_shapes) {
  for (auto [r, c] : std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {1, 50}, {50, 1}, {33, 65}, {128, 96}, {64, 64}})
    check_shape<float>(r, c);
}

S2S_TEST(int8_shapes) {
  for (auto [r, c] : std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {17, 3}, {33, 65}, {256, 128}})
    check_shape<int8_t>(r, c);
}

S2S_TEST_MAIN()
