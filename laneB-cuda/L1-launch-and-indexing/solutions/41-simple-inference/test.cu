#include <array>
#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(linear_layer) {
  for (auto s : {std::array{1, 64, 10}, std::array{7, 33, 129}, std::array{128, 256, 512}}) {
    const int B = s[0], I = s[1], O = s[2];
    auto x = s2s::random_vec<float>(size_t(B) * I, -1, 1, 1), W = s2s::random_vec<float>(size_t(O) * I, -1, 1, 2),
         b = s2s::random_vec<float>(size_t(O), -1, 1, 3);
    s2s::DeviceBuffer<float> dx(x), dW(W), db(b), dy(size_t(B) * O);
    solve(dx.get(), dW.get(), db.get(), dy.get(), B, I, O);
    CUDA_CHECK_LAUNCH();
    auto y = dy.download();
    std::vector<double> want(size_t(B) * O);
    for (int r = 0; r < B; ++r)
      for (int o = 0; o < O; ++o) {
        double acc = b[size_t(o)];
        for (int i = 0; i < I; ++i) acc += double(x[size_t(r) * I + i]) * W[size_t(o) * I + i];
        want[size_t(r) * O + o] = acc;
      }
    CHECK_ALLCLOSE(y.data(), want.data(), want.size(), 1e-4, 1e-5 * std::sqrt(double(I)));
  }
}

int main() { return s2s::run_all(); }
