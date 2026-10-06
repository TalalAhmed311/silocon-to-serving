#include <array>
#include <cmath>
#include <cstring>
#include <random>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int M, int N, int K) {
  auto a = s2s::random_vec<float>(size_t(M) * N, -1, 1, unsigned(M + N)), b = s2s::random_vec<float>(size_t(N) * K, -1, 1, unsigned(K));
  s2s::DeviceBuffer<float> da(a), db(b), dc(size_t(M) * K);
  solve(da.get(), db.get(), dc.get(), M, N, K);
  CUDA_CHECK_LAUNCH();
  auto c = dc.download();
  std::vector<double> want(size_t(M) * K, 0.0);
  for (int r = 0; r < M; ++r)
    for (int i = 0; i < N; ++i)
      for (int k = 0; k < K; ++k) want[size_t(r) * K + k] += double(a[size_t(r) * N + i]) * b[size_t(i) * K + k];
  CHECK_ALLCLOSE(c.data(), want.data(), want.size(), 1e-4, 1e-5 * std::sqrt(double(N)));
}

S2S_TEST(edge_shapes) {
  for (auto s : {std::array{1, 1, 1}, std::array{1, 7, 1}, std::array{17, 1, 33}, std::array{64, 64, 64}}) check(s[0], s[1], s[2]);
}

S2S_TEST(exit_check_20_random_shapes) {
  std::mt19937 rng(2024);
  for (int t = 0; t < 20; ++t) check(int(rng() % 300 + 1), int(rng() % 300 + 1), int(rng() % 300 + 1));
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 2048;
  s2s::DeviceBuffer<float> a(size_t(n) * n), b(size_t(n) * n), c(size_t(n) * n);
  a.zero(); b.zero();
  s2s::table_header("GFLOP/s");
  auto t = s2s::time_gpu([&] { solve(a.get(), b.get(), c.get(), n, n, n); }, 2, 10);
  s2s::report("matmul_tiled", "16x16 smem tiles", n, t, 2.0 * n * n * n / (t.median_ms * 1e6), "GFLOP/s", 0);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
