#include <algorithm>
#include <cmath>

#include <d4/attention.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

using h16 = __half;
constexpr int D = d4::AD;

static std::vector<h16> to_h(const std::vector<float>& v) { std::vector<h16> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = __float2half(v[i]); return o; }
static std::vector<float> to_f(const std::vector<h16>& v) { std::vector<float> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = __half2float(v[i]); return o; }

// Reference in double from the fp16-rounded inputs.
static std::vector<float> ref_attn(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int BH, int N, bool causal) {
  std::vector<float> o(size_t(BH) * N * D);
  const double sc = 1.0 / std::sqrt(double(D));
  std::vector<double> s(size_t(N));
  for (int bh = 0; bh < BH; ++bh)
    for (int i = 0; i < N; ++i) {
      double m = -1e300;
      const int jmax = causal ? i : N - 1;
      for (int j = 0; j <= jmax; ++j) {
        double a = 0;
        for (int d = 0; d < D; ++d) a += double(q[(size_t(bh) * N + i) * D + d]) * k[(size_t(bh) * N + j) * D + d];
        s[size_t(j)] = a * sc;
        m = std::max(m, s[size_t(j)]);
      }
      double l = 0;
      for (int j = 0; j <= jmax; ++j) { s[size_t(j)] = std::exp(s[size_t(j)] - m); l += s[size_t(j)]; }
      for (int d = 0; d < D; ++d) {
        double a = 0;
        for (int j = 0; j <= jmax; ++j) a += s[size_t(j)] * v[(size_t(bh) * N + j) * D + d];
        o[(size_t(bh) * N + i) * D + d] = float(a / l);
      }
    }
  return o;
}

static void check_all(int BH, int N, bool causal) {
  const size_t n = size_t(BH) * N * D;
  auto qh = to_h(s2s::random_vec<float>(n, -1, 1, 1)), kh = to_h(s2s::random_vec<float>(n, -1, 1, 2)), vh = to_h(s2s::random_vec<float>(n, -1, 1, 3));
  auto want = ref_attn(to_f(qh), to_f(kh), to_f(vh), BH, N, causal);
  s2s::DeviceBuffer<h16> Q(qh), K(kh), V(vh), O(n);
  // fp16 output: one rounding (u = 2^-11 ≈ 4.9e-4) on values |o| ≤ 1, plus __expf and fp32 sums → 2e-3 abs.
  auto check = [&](const char*) { auto got = to_f(O.download()); CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3); };
  s2s::DeviceBuffer<float> S(size_t(BH) * N * N);
  d4::attn_naive(Q.get(), K.get(), V.get(), O.get(), S.get(), BH, N, causal);
  CUDA_CHECK_LAUNCH();
  check("naive");
  s2s::DeviceBuffer<float> Oacc(n), M(size_t(BH) * N), L(size_t(BH) * N);
  d4::attn_flash1(Q.get(), K.get(), V.get(), O.get(), Oacc.get(), M.get(), L.get(), BH, N, causal);
  CUDA_CHECK_LAUNCH();
  check("fa1");
  d4::attn_flash2(Q.get(), K.get(), V.get(), O.get(), BH, N, causal);
  CUDA_CHECK_LAUNCH();
  check("fa2");
}

S2S_TEST(full) { check_all(2, 256, false); check_all(1, 100, false); }
S2S_TEST(causal) { check_all(2, 256, true); check_all(3, 77, true); }

S2S_TEST(paged_decode) {
  const int B = 3, H = 8, Hkv = 2, bs = 16, max_blocks = 8, num_blocks = 32;
  const std::vector<int> lens = {1, 37, 128};
  auto qf = s2s::random_vec<float>(size_t(B) * H * D, -1, 1, 4);
  auto kf = s2s::random_vec<float>(size_t(num_blocks) * bs * Hkv * D, -1, 1, 5), vf = s2s::random_vec<float>(kf.size(), -1, 1, 6);
  auto qh = to_h(qf), kh = to_h(kf), vh = to_h(vf);
  qf = to_f(qh); kf = to_f(kh); vf = to_f(vh);
  // a shuffled block table: logical block i of seq b lives at a "random" physical block
  std::vector<int> table(size_t(B) * max_blocks);
  for (int b = 0; b < B; ++b) for (int i = 0; i < max_blocks; ++i) table[size_t(b) * max_blocks + i] = (b * 11 + i * 5 + 3) % num_blocks;
  std::vector<float> want(size_t(B) * H * D);
  const double sc = 1.0 / std::sqrt(double(D));
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int kvh = h / (H / Hkv), len = lens[size_t(b)];
      std::vector<double> s(size_t(len));
      double m = -1e300, l = 0;
      for (int t = 0; t < len; ++t) {
        const int phys = table[size_t(b) * max_blocks + t / bs];
        const size_t kv = ((size_t(phys) * bs + t % bs) * Hkv + kvh) * D;
        double a = 0;
        for (int d = 0; d < D; ++d) a += double(qf[(size_t(b) * H + h) * D + d]) * kf[kv + d];
        s[size_t(t)] = a * sc;
        m = std::max(m, s[size_t(t)]);
      }
      for (int t = 0; t < len; ++t) { s[size_t(t)] = std::exp(s[size_t(t)] - m); l += s[size_t(t)]; }
      for (int d = 0; d < D; ++d) {
        double a = 0;
        for (int t = 0; t < len; ++t) {
          const int phys = table[size_t(b) * max_blocks + t / bs];
          a += s[size_t(t)] * vf[((size_t(phys) * bs + t % bs) * Hkv + kvh) * D + d];
        }
        want[(size_t(b) * H + h) * D + d] = float(a / l);
      }
    }
  s2s::DeviceBuffer<h16> q(qh), kc(kh), vc(vh), out(want.size());
  s2s::DeviceBuffer<int> bt(table), sl(lens);
  d4::paged_decode_attention(q.get(), kc.get(), vc.get(), bt.get(), sl.get(), out.get(), B, H, Hkv, bs, max_blocks);
  CUDA_CHECK_LAUNCH();
  auto got = to_f(out.download());
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST_MAIN()
