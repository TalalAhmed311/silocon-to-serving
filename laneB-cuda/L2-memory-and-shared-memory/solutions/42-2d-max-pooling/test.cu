#include <algorithm>
#include <array>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(windows) {
  for (auto c : {std::array{4, 4, 2, 2}, std::array{9, 13, 3, 1}, std::array{224, 224, 3, 2}, std::array{7, 7, 7, 7}}) {
    const int H = c[0], W = c[1], k = c[2], s = c[3], OH = (H - k) / s + 1, OW = (W - k) / s + 1;
    auto in = s2s::random_vec<float>(size_t(H) * W, -5, -1, unsigned(H * W));   // all negative: catches init-to-0 bugs
    s2s::DeviceBuffer<float> di(in), dout(size_t(OH) * OW);
    solve(di.get(), dout.get(), H, W, k, s);
    CUDA_CHECK_LAUNCH();
    std::vector<float> want(size_t(OH) * OW, -1e30f);
    for (int y = 0; y < OH; ++y)
      for (int x = 0; x < OW; ++x)
        for (int i = 0; i < k; ++i)
          for (int j = 0; j < k; ++j) want[size_t(y) * OW + x] = std::max(want[size_t(y) * OW + x], in[size_t(y * s + i) * W + x * s + j]);
    auto got = dout.download();
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
  }
}

int main() { return s2s::run_all(); }
