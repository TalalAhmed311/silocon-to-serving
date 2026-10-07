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

static void check(int N, int d, int H) {
  auto q = s2s::random_vec<float>(size_t(H) * N * d, -1, 1, 1), k = s2s::random_vec<float>(q.size(), -1, 1, 2), v = s2s::random_vec<float>(q.size(), -1, 1, 3);
  attnref::Cfg c; c.H = c.Hkv = H; c.Lq = c.Lk = N; c.d = d; c.causal = true;
  for (int h = 0; h < H; ++h) c.alibi.push_back(std::pow(2.f, -8.f * (h + 1) / H));
  s2s::DeviceBuffer<float> Q(q), K(k), V(v), O(q.size());
  solve(Q.get(), K.get(), V.get(), O.get(), N, d, H);
  CUDA_CHECK_LAUNCH();
  expect_close(O.download(), attnref::run(c, q, k, v));
}
S2S_TEST(shapes) { check(64, 64, 8); check(150, 32, 4); }

S2S_TEST_MAIN()
