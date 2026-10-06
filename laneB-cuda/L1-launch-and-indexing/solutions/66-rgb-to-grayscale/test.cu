#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(shapes) {
  for (auto wh : {std::pair{1, 1}, std::pair{7, 3}, std::pair{1920, 1080}}) {
    const int n = wh.first * wh.second;
    auto in = s2s::random_vec<float>(size_t(3 * n), 0, 1, unsigned(n));
    s2s::DeviceBuffer<float> din(in), dout(size_t(n));
    solve(din.get(), dout.get(), wh.first, wh.second);
    CUDA_CHECK_LAUNCH();
    auto out = dout.download();
    std::vector<float> want(size_t(n));
    for (int p = 0; p < n; ++p) want[size_t(p)] = kR * in[size_t(3 * p)] + kG * in[size_t(3 * p + 1)] + kB * in[size_t(3 * p + 2)];
    CHECK_ALLCLOSE(out.data(), want.data(), want.size(), 1e-6, 1e-6);  // FMA contraction may differ by 1 ulp
  }
}

int main() { return s2s::run_all(); }
