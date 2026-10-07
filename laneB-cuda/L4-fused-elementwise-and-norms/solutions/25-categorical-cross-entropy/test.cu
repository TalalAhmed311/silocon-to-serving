#include <algorithm>
#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int C, float scale) {
  auto lg = s2s::random_vec<float>(size_t(N) * C, -scale, scale, unsigned(N + C));
  auto lb = s2s::random_vec<int>(size_t(N), 0, C - 1, 7);
  double want = 0;
  for (int r = 0; r < N; ++r) {
    double m = -1e300, s = 0;
    for (int c = 0; c < C; ++c) m = std::max(m, double(lg[size_t(r) * C + c]));
    for (int c = 0; c < C; ++c) s += std::exp(lg[size_t(r) * C + c] - m);
    want += m + std::log(s) - lg[size_t(r) * C + size_t(lb[size_t(r)])];
  }
  want /= N;
  s2s::DeviceBuffer<float> dl(lg), out(1);
  s2s::DeviceBuffer<int> dlb(lb);
  solve(dl.get(), dlb.get(), out.get(), N, C);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(out.download()[0], want, 1e-4, 1e-5);
}

S2S_TEST(shapes) { check(1, 2, 1); check(100, 10, 3); check(64, 32000, 5); }
S2S_TEST(confident_logits) { check(32, 1000, 80); }   // naive −log(softmax) underflows to −log(0) here

S2S_TEST_MAIN()
