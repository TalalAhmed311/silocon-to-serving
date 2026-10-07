#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(correct) {
  const int R = 256, C = 512;
  auto in = s2s::random_vec<int>(size_t(R) * C, -50, 50, 3);
  std::vector<float> f(in.begin(), in.end()), want(size_t(R) * (C / 32), 0.f);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) want[size_t(r) * (C / 32) + c / 32] += f[size_t(r) * C + c];
  s2s::DeviceBuffer<float> din(f), dout(want.size());
  tile_rowsum(din.get(), dout.get(), R, C);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);   // small ints: exact
}

S2S_TEST(bench) {
  if (!g_bench) return;
  const int R = 8192, C = 8192;
  s2s::DeviceBuffer<float> in(size_t(R) * C), out(size_t(R) * (C / 32));
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { tile_rowsum(in.get(), out.get(), R, C); });
  s2s::report("tile_rowsum", "yours", double(R) * C, t, 4.0 * R * C / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs() / 2);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
