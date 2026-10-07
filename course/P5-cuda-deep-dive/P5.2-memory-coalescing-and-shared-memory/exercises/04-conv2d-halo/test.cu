#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int H, int W, int K) {
  auto in = s2s::random_vec<float>(size_t(H) * W, -1, 1, unsigned(H + W + K));
  auto w = s2s::random_vec<float>(size_t(K) * K, -1, 1, 9);
  std::vector<float> want(in.size(), 0.f);
  const int R = K / 2;
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      double acc = 0;
      for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) {
          const int rr = r + i - R, cc = c + j - R;
          if (rr >= 0 && rr < H && cc >= 0 && cc < W) acc += double(in[size_t(rr) * W + cc]) * w[size_t(i) * K + j];
        }
      want[size_t(r) * W + c] = float(acc);
    }
  s2s::DeviceBuffer<float> din(in), dout(in.size());
  dout.zero();
  conv2d(din.get(), w.data(), dout.get(), H, W, K);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-5, 1e-5);
}

S2S_TEST(shapes) { check(1, 1, 3); check(17, 33, 3); check(64, 64, 5); check(100, 37, 7); check(512, 512, 1); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int H = 4096, W = 4096, K = 7;
  s2s::DeviceBuffer<float> in(size_t(H) * W), out(size_t(H) * W);
  auto w = s2s::random_vec<float>(49);
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { conv2d(in.get(), w.data(), out.get(), H, W, K); });
  s2s::report("conv2d", "yours (K=7)", double(H) * W, t, 8.0 * H * W / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
