#include <s2s/check.hpp>
#include "../_shared/attn_ref.hpp"
#include "kernel.cu"
#include "s2s_cuda.cuh"

// fp32 everywhere; the GPU uses __expf and a different summation order → rtol/atol 1e-4.
static void expect_close(const std::vector<float>& got, const std::vector<float>& want) {
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

// [L, H, dh] (packed heads) <-> [H, L, dh]
[[maybe_unused]] static std::vector<float> unpack(const std::vector<float>& x, int L, int H, int dh) {
  std::vector<float> y(x.size());
  for (int i = 0; i < L; ++i) for (int h = 0; h < H; ++h) for (int t = 0; t < dh; ++t) y[(size_t(h) * L + i) * dh + t] = x[(size_t(i) * H + h) * dh + t];
  return y;
}

static void check(int N, int dm, int h) {
  auto q = s2s::random_vec<float>(size_t(N) * dm, -1, 1, 1), k = s2s::random_vec<float>(size_t(N) * dm, -1, 1, 2), v = s2s::random_vec<float>(size_t(N) * dm, -1, 1, 3);
  const int dh = dm / h;
  attnref::Cfg c; c.H = c.Hkv = h; c.Lq = c.Lk = N; c.d = dh;
  auto want_hld = attnref::run(c, unpack(q, N, h, dh), unpack(k, N, h, dh), unpack(v, N, h, dh));
  s2s::DeviceBuffer<float> Q(q), K(k), V(v), O(size_t(N) * dm);
  solve(Q.get(), K.get(), V.get(), O.get(), N, dm, h);
  CUDA_CHECK_LAUNCH();
  expect_close(unpack(O.download(), N, h, dh), want_hld);
}
S2S_TEST(shapes) { check(1, 8, 2); check(64, 256, 4); check(77, 512, 8); }

S2S_TEST_MAIN()
