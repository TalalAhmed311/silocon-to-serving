#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(volume) {
  const int D = 12, H = 20, W = 33, KD = 3, KH = 5, KW = 3, OD = D - KD + 1, OH = H - KH + 1, OW = W - KW + 1;
  auto in = s2s::random_vec<float>(size_t(D) * H * W, -1, 1, 1), k = s2s::random_vec<float>(size_t(KD) * KH * KW, -1, 1, 2);
  s2s::DeviceBuffer<float> di(in), dk(k), dout(size_t(OD) * OH * OW);
  solve(di.get(), dk.get(), dout.get(), D, H, W, KD, KH, KW);
  CUDA_CHECK_LAUNCH();
  std::vector<double> want(size_t(OD) * OH * OW);
  for (int z = 0; z < OD; ++z)
    for (int y = 0; y < OH; ++y)
      for (int x = 0; x < OW; ++x) {
        double s = 0;
        for (int a = 0; a < KD; ++a)
          for (int b = 0; b < KH; ++b)
            for (int c = 0; c < KW; ++c) s += double(in[(size_t(z + a) * H + y + b) * W + x + c]) * k[size_t((a * KH + b) * KW + c)];
        want[(size_t(z) * OH + y) * OW + x] = s;
      }
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

int main() { return s2s::run_all(); }
