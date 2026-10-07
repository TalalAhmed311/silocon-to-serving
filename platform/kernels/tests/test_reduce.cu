#include <cmath>

#include <d4/reduce.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

static void check(int rung, long long n) {
  auto h = s2s::random_vec<float>(size_t(n), -1, 1, unsigned(n));
  double want = 0, mag = 0;
  for (float x : h) { want += x; mag += std::fabs(x); }
  s2s::DeviceBuffer<float> in(h), out(1), partial(size_t(d4::ceil_div(n, d4::RED_THREADS)) + 1);
  d4::reduce_sum(rung, in.get(), out.get(), partial.get(), n);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(out.download()[0], want, 0.0, 1e-6 * mag + 1e-5);
}

S2S_TEST(all_rungs) {
  for (int r = 1; r <= 6; ++r)
    for (long long n : {1LL, 255LL, 256LL, 513LL, 100000LL, 1LL << 22}) check(r, n);
}

S2S_TEST_MAIN()
