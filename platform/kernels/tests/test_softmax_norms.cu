#include <algorithm>
#include <cmath>

#include <d4/norms.cuh>
#include <d4/softmax.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

static std::vector<float> ref_softmax(const std::vector<float>& x, int rows, int cols) {
  std::vector<float> y(x.size());
  for (int r = 0; r < rows; ++r) {
    double m = -1e300, s = 0;
    for (int c = 0; c < cols; ++c) m = std::max(m, double(x[size_t(r) * cols + c]));
    for (int c = 0; c < cols; ++c) s += std::exp(double(x[size_t(r) * cols + c]) - m);
    for (int c = 0; c < cols; ++c) y[size_t(r) * cols + c] = float(std::exp(double(x[size_t(r) * cols + c]) - m) / s);
  }
  return y;
}

S2S_TEST(softmax_versions) {
  for (auto [rows, cols] : {std::pair{1, 1}, std::pair{7, 33}, std::pair{64, 1000}, std::pair{8, 4096}, std::pair{3, 50000}}) {
    auto x = s2s::random_vec<float>(size_t(rows) * cols, -20, 20, unsigned(rows * cols));
    auto want = ref_softmax(x, rows, cols);
    for (int v = 1; v <= 3; ++v) {
      s2s::DeviceBuffer<float> dx(x), dy(x.size());
      d4::softmax(v, dx.get(), dy.get(), rows, cols);
      CUDA_CHECK_LAUNCH();
      auto got = dy.download();
      CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-7);   // __expf: ~2 ulp; derived in P5.5 §3
    }
  }
}

template <class T>
static void check_rmsnorm(int rows, int cols, double rtol, double atol) {
  auto xf = s2s::random_vec<float>(size_t(rows) * cols, -3, 3, 5), wf = s2s::random_vec<float>(size_t(cols), 0.5, 1.5, 6);
  auto rf = s2s::random_vec<float>(size_t(rows) * cols, -1, 1, 7);
  // round inputs to T first, so the reference sees exactly what the GPU sees
  std::vector<T> x(xf.size()), w(wf.size()), r(rf.size());
  auto rt = [](float v) { if constexpr (std::is_same_v<T, float>) return v; else return T(v); };
  auto tf = [](T v) { if constexpr (std::is_same_v<T, float>) return v; else return float(v); };
  for (size_t i = 0; i < xf.size(); ++i) { x[i] = rt(xf[i]); r[i] = rt(rf[i]); }
  for (size_t i = 0; i < wf.size(); ++i) w[i] = rt(wf[i]);
  std::vector<float> want(xf.size()), want_fused(xf.size()), want_resid(xf.size());
  for (int rr = 0; rr < rows; ++rr) {
    double ss = 0, ss2 = 0;
    for (int c = 0; c < cols; ++c) {
      const double v = tf(x[size_t(rr) * cols + c]);
      ss += v * v;
      const float sum = tf(rt(tf(x[size_t(rr) * cols + c]) + tf(r[size_t(rr) * cols + c])));
      want_resid[size_t(rr) * cols + c] = sum;
      ss2 += double(sum) * sum;
    }
    const double inv = 1 / std::sqrt(ss / cols + 1e-6), inv2 = 1 / std::sqrt(ss2 / cols + 1e-6);
    for (int c = 0; c < cols; ++c) {
      want[size_t(rr) * cols + c] = float(tf(x[size_t(rr) * cols + c]) * inv * tf(w[size_t(c)]));
      want_fused[size_t(rr) * cols + c] = float(want_resid[size_t(rr) * cols + c] * inv2 * tf(w[size_t(c)]));
    }
  }
  s2s::DeviceBuffer<T> dx(x), dw(w), dy(x.size()), dr(r);
  d4::rmsnorm(dx.get(), dw.get(), dy.get(), rows, cols);
  CUDA_CHECK_LAUNCH();
  auto y = dy.download();
  std::vector<float> yf(y.size());
  for (size_t i = 0; i < y.size(); ++i) yf[i] = tf(y[i]);
  CHECK_ALLCLOSE(yf.data(), want.data(), want.size(), rtol, atol);
  d4::fused_add_rmsnorm(dx.get(), dr.get(), dw.get(), dy.get(), rows, cols);
  CUDA_CHECK_LAUNCH();
  auto y2 = dy.download(), r2 = dr.download();
  std::vector<float> y2f(y2.size()), r2f(r2.size());
  for (size_t i = 0; i < y2.size(); ++i) { y2f[i] = tf(y2[i]); r2f[i] = tf(r2[i]); }
  CHECK_ALLCLOSE(r2f.data(), want_resid.data(), want_resid.size(), 0.0, 0.0);   // same rounding on both sides
  CHECK_ALLCLOSE(y2f.data(), want_fused.data(), want_fused.size(), rtol, atol);
}

S2S_TEST(rmsnorm_fp32) { check_rmsnorm<float>(64, 4096, 1e-5, 1e-6); check_rmsnorm<float>(5, 1001, 1e-5, 1e-6); }
// bf16: 8 significant bits → one rounding of the output is ≤ 2^-8 relative ≈ 3.9e-3; allow 1e-2 (P5.5 §3).
S2S_TEST(rmsnorm_bf16) { check_rmsnorm<__nv_bfloat16>(32, 4096, 1e-2, 1e-2); }
// fp16: 11 significant bits → ≤ 2^-11 ≈ 4.9e-4 per rounding; allow 2e-3.
S2S_TEST(rmsnorm_fp16) { check_rmsnorm<__half>(32, 4096, 2e-3, 2e-3); }

S2S_TEST_MAIN()
