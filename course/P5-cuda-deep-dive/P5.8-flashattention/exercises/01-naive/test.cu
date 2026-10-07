#include <cstring>

#include <s2s/check.hpp>
#include "attn_ref.hpp"
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int BH, int N, bool causal) {
  const size_t n = size_t(BH) * N * attnref::D;
  auto qh = attnref::to_h(s2s::random_vec<float>(n, -1, 1, 1)), kh = attnref::to_h(s2s::random_vec<float>(n, -1, 1, 2));
  auto vh = attnref::to_h(s2s::random_vec<float>(n, -1, 1, 3));
  auto want = attnref::attention(attnref::to_f(qh), attnref::to_f(kh), attnref::to_f(vh), BH, N, causal);
  s2s::DeviceBuffer<__half> Q(qh), K(kh), V(vh), O(n);
  O.zero();
  s2s::DeviceBuffer<float> S(size_t(BH) * N * N);
  attention(Q.get(), K.get(), V.get(), O.get(), S.get(), BH, N, causal);
  CUDA_CHECK_LAUNCH();
  auto got = attnref::to_f(O.download());
  // fp16 output (u = 2^-11) of values |o| <= 1 plus fp32 math with __expf: 2e-3 absolute (README derives it).
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST(full) { check(2, 128, false); check(1, 77, false); }
S2S_TEST(causal) { check(2, 128, true); check(3, 100, true); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int BH = 16, N = 4096;
  const size_t n = size_t(BH) * N * attnref::D;
  s2s::DeviceBuffer<__half> Q(n), K(n), V(n), O(n);
  Q.zero(); K.zero(); V.zero();
  const bool causal = false;
  s2s::DeviceBuffer<float> S(size_t(BH) * N * N);
  auto t = s2s::time_gpu([&] { attention(Q.get(), K.get(), V.get(), O.get(), S.get(), BH, N, causal); }, 1, 5);
  std::printf("BH=%d N=%d: %.2f ms, %.2f TFLOP/s\n", BH, N, t.median_ms, 4.0 * BH * N * double(N) * attnref::D / (t.median_ms * 1e9));
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
