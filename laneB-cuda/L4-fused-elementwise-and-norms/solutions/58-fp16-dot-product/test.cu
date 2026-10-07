#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int n) {
  auto fa = s2s::random_vec<float>(size_t(n), -1, 1, 1), fb = s2s::random_vec<float>(size_t(n), -1, 1, 2);
  std::vector<__half> a(fa.size()), b(fb.size());
  double want = 0, mag = 0;
  for (int i = 0; i < n; ++i) {
    a[size_t(i)] = __float2half(fa[size_t(i)]); b[size_t(i)] = __float2half(fb[size_t(i)]);
    const double p = double(__half2float(a[size_t(i)])) * __half2float(b[size_t(i)]);
    want += p; mag += std::fabs(p);
  }
  s2s::DeviceBuffer<__half> da(a), db(b);
  s2s::DeviceBuffer<float> out(1);
  solve(da.get(), db.get(), out.get(), n);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(out.download()[0], want, 0.0, 1e-6 * mag + 1e-5);    // exact products, fp32 sums in a different order
}

S2S_TEST(sizes) { check(1); check(2); check(1001); check(1 << 22); }

// Accumulating in fp16 instead would lose everything past ~2048 equal terms (fp16 spacing at 2048 is 2):
S2S_TEST(why_fp32_accumulation) {
  const int n = 100000;
  std::vector<__half> one(size_t(n), __float2half(1.f));
  s2s::DeviceBuffer<__half> a(one), b(one);
  s2s::DeviceBuffer<float> out(1);
  solve(a.get(), b.get(), out.get(), n);
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(out.download()[0], float(n));
}

S2S_TEST_MAIN()
