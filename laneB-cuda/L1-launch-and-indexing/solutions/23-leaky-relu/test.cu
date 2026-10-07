#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;
static float g_alpha = 0.01f; static const float alpha = g_alpha;
static float ref(float x) { return x > 0 ? x : g_alpha * x; }

S2S_TEST(sizes) {
  for (int n : {1, 31, 256, 1000, 1 << 20, 5000003}) {
    auto in = s2s::random_vec<float>(size_t(n), -8, 8, unsigned(n));
    s2s::DeviceBuffer<float> din(in), dout(size_t(n));
    solve(din.get(), dout.get(), n, alpha);
    CUDA_CHECK_LAUNCH();
    auto out = dout.download();
    std::vector<float> want(size_t(n));
    for (int i = 0; i < n; ++i) want[size_t(i)] = ref(in[size_t(i)]);
    CHECK_ALLCLOSE(out.data(), want.data(), want.size(), 0.0, 1e-6);
  }
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 1 << 26;
  s2s::DeviceBuffer<float> a(size_t(n)), b(size_t(n));
  a.zero();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(a.get(), b.get(), n, alpha); });
  s2s::report("l1_23", "grid-stride", n, t, 8.0 * n / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
