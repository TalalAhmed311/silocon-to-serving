#include <array>
#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(shapes) {
  for (auto s : {std::array{5, 5, 3, 3}, std::array{37, 61, 5, 7}, std::array{512, 512, 9, 9}, std::array{100, 3, 1, 3}}) {
    const int R = s[0], C = s[1], KR = s[2], KC = s[3], OR = R - KR + 1, OC = C - KC + 1;
    auto in = s2s::random_vec<float>(size_t(R) * C, -1, 1, 1), k = s2s::random_vec<float>(size_t(KR) * KC, -1, 1, 2);
    s2s::DeviceBuffer<float> din(in), dk(k), dout(size_t(OR) * OC);
    solve(din.get(), dk.get(), dout.get(), R, C, KR, KC);
    CUDA_CHECK_LAUNCH();
    auto got = dout.download();
    std::vector<double> want(size_t(OR) * OC);
    for (int r = 0; r < OR; ++r)
      for (int c = 0; c < OC; ++c) {
        double acc = 0;
        for (int i = 0; i < KR; ++i)
          for (int j = 0; j < KC; ++j) acc += double(in[size_t(r + i) * C + c + j]) * k[size_t(i) * KC + j];
        want[size_t(r) * OC + c] = acc;
      }
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-5 * std::sqrt(double(KR * KC)));
  }
}

int main() { return s2s::run_all(); }
