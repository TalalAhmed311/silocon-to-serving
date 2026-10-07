#include <s2s/check.hpp>
#include "attn_ref.hpp"
#include "kernel.cuh"
#include "s2s_cuda.cuh"

// Build the paged cache from contiguous per-sequence K/V, run the kernel, and compare with the dense reference for the
// LAST query position of each sequence (decode = attention of one new query over the whole cached history).
static void check(const std::vector<int>& lens, int H, int Hkv, int bs) {
  const int B = int(lens.size()), D = attnref::D;
  int max_len = 0;
  for (int l : lens) max_len = std::max(max_len, l);
  const int max_blocks = (max_len + bs - 1) / bs, num_blocks = B * max_blocks + 3;
  std::vector<int> table(size_t(B) * max_blocks, 0);
  int next = num_blocks - 1;                                  // allocate physical blocks from the top, in reverse
  for (int b = 0; b < B; ++b) for (int i = 0; i < (lens[size_t(b)] + bs - 1) / bs; ++i) table[size_t(b) * max_blocks + i] = next--;
  auto kc = attnref::to_h(s2s::random_vec<float>(size_t(num_blocks) * bs * Hkv * D, -1, 1, 5));
  auto vc = attnref::to_h(s2s::random_vec<float>(kc.size(), -1, 1, 6));
  auto qh = attnref::to_h(s2s::random_vec<float>(size_t(B) * H * D, -1, 1, 4));
  auto kf = attnref::to_f(kc), vf = attnref::to_f(vc), qf = attnref::to_f(qh);
  std::vector<float> want(size_t(B) * H * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const int len = lens[size_t(b)], kvh = h / (H / Hkv);
      // gather this (b, kvh) history into dense [len, D] and reuse the dense reference with N = len, query = last row
      std::vector<float> q(size_t(len) * D, 0.f), k(size_t(len) * D), v(size_t(len) * D);
      for (int t = 0; t < len; ++t) {
        const size_t src = ((size_t(table[size_t(b) * max_blocks + t / bs]) * bs + t % bs) * Hkv + kvh) * D;
        for (int d = 0; d < D; ++d) { k[size_t(t) * D + d] = kf[src + d]; v[size_t(t) * D + d] = vf[src + d]; }
      }
      for (int d = 0; d < D; ++d) q[size_t(len - 1) * D + d] = qf[(size_t(b) * H + h) * D + d];
      auto o = attnref::attention(q, k, v, 1, len, true);       // causal: the last row sees everything
      for (int d = 0; d < D; ++d) want[(size_t(b) * H + h) * D + d] = o[size_t(len - 1) * D + d];
    }
  s2s::DeviceBuffer<__half> q(qh), K(kc), V(vc), out(want.size());
  s2s::DeviceBuffer<int> bt(table), sl(lens);
  paged_decode(q.get(), K.get(), V.get(), bt.get(), sl.get(), out.get(), B, H, Hkv, bs, max_blocks);
  CUDA_CHECK_LAUNCH();
  auto got = attnref::to_f(out.download());
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST(mha) { check({1, 15, 16, 17, 200}, 4, 4, 16); }
S2S_TEST(gqa) { check({33, 512}, 8, 2, 16); check({7}, 32, 8, 32); }

S2S_TEST_MAIN()
