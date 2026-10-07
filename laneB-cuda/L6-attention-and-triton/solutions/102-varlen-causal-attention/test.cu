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

static void check(std::vector<int> lens, int H, int d) {
  const int B = int(lens.size());
  std::vector<int> cu{0};
  int maxl = 0;
  for (int l : lens) { cu.push_back(cu.back() + l); maxl = std::max(maxl, l); }
  const int T = cu.back();
  auto q = s2s::random_vec<float>(size_t(T) * H * d, -1, 1, 1), k = s2s::random_vec<float>(q.size(), -1, 1, 2), v = s2s::random_vec<float>(q.size(), -1, 1, 3);
  std::vector<float> want(q.size());
  for (int b = 0; b < B; ++b) {                       // reference: each sequence alone, causal
    const int L = lens[size_t(b)];
    auto slice = [&](const std::vector<float>& x) { return unpack(std::vector<float>(x.begin() + size_t(cu[size_t(b)]) * H * d, x.begin() + size_t(cu[size_t(b) + 1]) * H * d), L, H, d); };
    attnref::Cfg c; c.H = c.Hkv = H; c.Lq = c.Lk = L; c.d = d; c.causal = true;
    auto o = attnref::run(c, slice(q), slice(k), slice(v));     // [H, L, d]
    for (int h = 0; h < H; ++h) for (int i = 0; i < L; ++i) for (int t = 0; t < d; ++t)
      want[(size_t(cu[size_t(b)] + i) * H + h) * d + t] = o[(size_t(h) * L + i) * d + t];
  }
  s2s::DeviceBuffer<float> Q(q), K(k), V(v), O(q.size());
  s2s::DeviceBuffer<int> CU(cu);
  solve(Q.get(), K.get(), V.get(), O.get(), CU.get(), B, maxl, H, d);
  CUDA_CHECK_LAUNCH();
  expect_close(O.download(), want);
}
S2S_TEST(varlen) { check({1}, 2, 32); check({5, 130, 64, 1}, 4, 64); check({200, 3}, 2, 32); }

S2S_TEST_MAIN()
