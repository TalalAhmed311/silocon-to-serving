#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static std::vector<float> gaussian(int K, double sigma) {
  std::vector<float> g(size_t(K) * K);
  double s = 0;
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < K; ++j) { double v = std::exp(-((i - K / 2) * (i - K / 2) + (j - K / 2) * (j - K / 2)) / (2 * sigma * sigma)); g[size_t(i) * K + j] = float(v); s += v; }
  for (auto& v : g) v = float(v / s);
  return g;
}

S2S_TEST(same_size_zero_padding) {
  for (auto hw : {std::pair{1, 1}, std::pair{17, 33}, std::pair{480, 640}}) {
    const int H = hw.first, W = hw.second, K = 5;
    auto img = s2s::random_vec<float>(size_t(H) * W, 0, 1, 7), g = gaussian(K, 1.2);
    s2s::DeviceBuffer<float> di(img), dg(g), dout(size_t(H) * W);
    solve(di.get(), dg.get(), dout.get(), H, W, K);
    CUDA_CHECK_LAUNCH();
    std::vector<double> want(size_t(H) * W);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        double s = 0;
        for (int i = 0; i < K; ++i)
          for (int j = 0; j < K; ++j) {
            int yy = y + i - K / 2, xx = x + j - K / 2;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) s += double(img[size_t(yy) * W + xx]) * g[size_t(i) * K + j];
          }
        want[size_t(y) * W + x] = s;
      }
    auto got = dout.download();
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-5, 1e-6);
  }
}

int main() { return s2s::run_all(); }
