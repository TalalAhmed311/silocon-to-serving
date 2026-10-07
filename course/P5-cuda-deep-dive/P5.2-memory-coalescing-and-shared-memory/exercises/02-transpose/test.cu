#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int R, int C) {
  auto in = s2s::random_vec<float>(size_t(R) * C, -1, 1, unsigned(R * 31 + C));
  std::vector<float> want(in.size());
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) want[size_t(c) * R + r] = in[size_t(r) * C + c];
  s2s::DeviceBuffer<float> din(in), dout(in.size());
  transpose(din.get(), dout.get(), R, C);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(shapes) {
  check(1, 1); check(32, 32); check(33, 65); check(1000, 7); check(7, 1000); check(1024, 2048);
}

// L2 exit check: ≥ 80% of measured copy bandwidth at 8192².
S2S_TEST(bench) {
  if (!g_bench) return;
  const int R = 8192, C = 8192;
  s2s::DeviceBuffer<float> in(size_t(R) * C), out(size_t(R) * C);
  const double copy = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { transpose(in.get(), out.get(), R, C); });
  const double gbs = 8.0 * R * C / (t.median_ms * 1e6);
  s2s::report("transpose", "yours", double(R) * C, t, gbs, "GB/s", copy);
  std::printf("exit check (>= 80%% of copy): %s\n", gbs >= 0.8 * copy ? "PASS" : "not yet");
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
