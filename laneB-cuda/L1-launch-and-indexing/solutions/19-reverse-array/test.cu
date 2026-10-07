#include <algorithm>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(sizes_odd_and_even) {
  for (int n : {1, 2, 3, 255, 256, 257, 1 << 20, (1 << 20) + 1}) {
    auto a = s2s::random_vec<float>(size_t(n), -1, 1, unsigned(n));
    s2s::DeviceBuffer<float> d(a);
    solve(d.get(), n);
    CUDA_CHECK_LAUNCH();
    auto got = d.download();
    std::reverse(a.begin(), a.end());
    CHECK_ALLCLOSE(got.data(), a.data(), a.size(), 0.0, 0.0);
  }
}

int main() { return s2s::run_all(); }
