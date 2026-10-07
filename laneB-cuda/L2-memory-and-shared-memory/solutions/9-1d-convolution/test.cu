#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(sizes) {
  for (auto nk : {std::pair{1, 1}, std::pair{10, 3}, std::pair{1000, 31}, std::pair{1 << 20, 255}, std::pair{5000, 2049}}) {
    const int N = nk.first, K = nk.second;
    auto in = s2s::random_vec<float>(size_t(N), -1, 1, 1), k = s2s::random_vec<float>(size_t(K), -1, 1, 2);
    s2s::DeviceBuffer<float> din(in), dk(k), dout(size_t(N - K + 1));
    solve(din.get(), dk.get(), dout.get(), N, K);
    CUDA_CHECK_LAUNCH();
    auto got = dout.download();
    std::vector<double> want(size_t(N - K + 1));
    for (int i = 0; i <= N - K; ++i) { double s = 0; for (int j = 0; j < K; ++j) s += double(in[size_t(i + j)]) * k[size_t(j)]; want[size_t(i)] = s; }
    CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-5 * std::sqrt(double(K)));
  }
}

int main() { return s2s::run_all(); }
