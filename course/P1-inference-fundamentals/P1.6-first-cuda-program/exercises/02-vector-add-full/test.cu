// Compares the GPU result with a CPU loop for sizes that are not multiples of the block size.
// A float add is exactly rounded on both sides, so the comparison is exact (rtol = atol = 0).
#include <vector>

#include <s2s/check.hpp>
#include <kernel.cuh>
#include "s2s_cuda.cuh"

static void check(int n) {
  auto a = s2s::random_vec<float>(size_t(n), -1, 1, 1), b = s2s::random_vec<float>(size_t(n), -1, 1, 2);
  std::vector<float> want(static_cast<size_t>(n)), got(static_cast<size_t>(n), -999.0f);  // -999 marks "never written"
  for (int i = 0; i < n; ++i) want[size_t(i)] = a[size_t(i)] + b[size_t(i)];
  vector_add(a.data(), b.data(), got.data(), n);
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(sizes) {
  for (int n : {1, 1000, 1 << 20, (1 << 20) + 3}) check(n);
}

S2S_TEST_MAIN()
