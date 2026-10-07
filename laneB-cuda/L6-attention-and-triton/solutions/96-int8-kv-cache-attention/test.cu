#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(std::vector<int> lens, int H, int Hkv, int L, int D) {
  const int B = int(lens.size());
  auto q = s2s::random_vec<float>(size_t(B) * H * D, -1, 1, 1);
  auto kqi = s2s::random_vec<int>(size_t(B) * Hkv * L * D, -127, 127, 2), vqi = s2s::random_vec<int>(kqi.size(), -127, 127, 3);
  std::vector<int8_t> kq(kqi.begin(), kqi.end()), vq(vqi.begin(), vqi.end());
  auto ks = s2s::random_vec<float>(size_t(B) * Hkv * L, 0.002, 0.02, 4), vs = s2s::random_vec<float>(ks.size(), 0.002, 0.02, 5);
  std::vector<float> want(q.size());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int kvh = h / (H / Hkv);
      const size_t base = (size_t(b) * Hkv + kvh) * L;
      std::vector<double> s(size_t(lens[size_t(b)]));
      double m = -1e300, l = 0;
      for (int j = 0; j < lens[size_t(b)]; ++j) {
        double a = 0;
        for (int t = 0; t < D; ++t) a += double(q[(size_t(b) * H + h) * D + t]) * kq[(base + j) * D + t];
        s[size_t(j)] = a * ks[base + j] / std::sqrt(double(D));
        m = std::max(m, s[size_t(j)]);
      }
      for (auto& x : s) { x = std::exp(x - m); l += x; }
      for (int t = 0; t < D; ++t) {
        double a = 0;
        for (int j = 0; j < lens[size_t(b)]; ++j) a += s[size_t(j)] * vs[base + j] * vq[(base + j) * D + t];
        want[(size_t(b) * H + h) * D + t] = float(a / l);
      }
    }
  s2s::DeviceBuffer<float> dq(q), dks(ks), dvs(vs), out(q.size());
  s2s::DeviceBuffer<int8_t> dk(kq), dv(vq);
  s2s::DeviceBuffer<int> dl(lens);
  solve(dq.get(), dk.get(), dv.get(), dks.get(), dvs.get(), dl.get(), out.get(), B, H, Hkv, L, D);
  CUDA_CHECK_LAUNCH();
  auto got = out.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(shapes) { check({1, 50}, 4, 4, 64, 64); check({300, 7, 128}, 8, 2, 300, 128); check({33}, 2, 1, 40, 32); }

S2S_TEST_MAIN()
