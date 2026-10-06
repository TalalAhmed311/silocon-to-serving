#include <algorithm>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int N) {
  auto p = s2s::random_vec<float>(size_t(N) * 3, -100, 100, unsigned(N));
  s2s::DeviceBuffer<float> dp(p);
  s2s::DeviceBuffer<int> di(size_t(N));
  solve(dp.get(), di.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = di.download();
  for (int i = 0; i < N; ++i) {          // compare distances (not indices): float rounding can reorder near-ties
    float best = 3.4e38f;
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      const float dx = p[3 * j] - p[3 * i], dy = p[3 * j + 1] - p[3 * i + 1], dz = p[3 * j + 2] - p[3 * i + 2];
      best = std::min(best, dx * dx + dy * dy + dz * dz);
    }
    const int g = got[size_t(i)];
    CHECK(g >= 0 && g < N && g != i);
    const float dx = p[3 * g] - p[3 * i], dy = p[3 * g + 1] - p[3 * i + 1], dz = p[3 * g + 2] - p[3 * i + 2];
    CHECK_NEAR(dx * dx + dy * dy + dz * dz, best, 1e-5, 1e-4);
  }
}

S2S_TEST(sizes) { for (int n : {2, 3, 255, 257, 2000}) check(n); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int N = 1 << 16;
  s2s::DeviceBuffer<float> p(s2s::random_vec<float>(size_t(N) * 3));
  s2s::DeviceBuffer<int> i(size_t(N));
  s2s::table_header("Gpairs/s");
  auto t = s2s::time_gpu([&] { solve(p.get(), i.get(), N); }, 2, 10);
  s2s::report("nearest_neighbor", "smem tiles, broadcast", N, t, double(N) * N / (t.median_ms * 1e6), "Gpairs/s", 0);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
