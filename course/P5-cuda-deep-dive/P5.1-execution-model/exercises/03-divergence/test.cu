#include <algorithm>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(correct) {
  for (int n : {1, 255, 1 << 20}) {
    auto in = s2s::random_vec<float>(size_t(n), -1, 2, unsigned(n));
    std::vector<float> want(in.size());
    for (size_t i = 0; i < in.size(); ++i) want[i] = std::min(std::max(in[i], 0.f), 1.f) * 3.f;
    s2s::DeviceBuffer<float> din(in), dout(in.size());
    clamp_scale(din.get(), dout.get(), n, 3.f);
    CUDA_CHECK_LAUNCH();
    auto got = dout.download();
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
  }
}

S2S_TEST(bench) {     // run with --bench before and after your fix; also compare `ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio`
  if (!g_bench) return;
  const int n = 1 << 24;
  s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(n), -1, 2)), out(size_t(n));
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { clamp_scale(in.get(), out.get(), n, 3.f); });
  s2s::report("clamp_scale", "yours", n, t, 8.0 * n / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
