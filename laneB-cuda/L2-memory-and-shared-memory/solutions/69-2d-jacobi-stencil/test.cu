#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(iterations) {
  for (int iters : {0, 1, 7, 50}) {
    const int H = 37, W = 53;
    auto g = s2s::random_vec<float>(size_t(H) * W, 0, 1, 9);
    s2s::DeviceBuffer<float> dg(g), dout(size_t(H) * W);
    solve(dg.get(), dout.get(), H, W, iters);
    CUDA_CHECK_LAUNCH();
    std::vector<float> u(g), v(g);
    for (int k = 0; k < iters; ++k) {
      for (int y = 1; y < H - 1; ++y)
        for (int x = 1; x < W - 1; ++x) { size_t i = size_t(y) * W + x; v[i] = 0.25f * (u[i - 1] + u[i + 1] + u[i - W] + u[i + W]); }
      std::swap(u, v);
    }
    auto got = dout.download();
    CHECK_ALLCLOSE(got.data(), u.data(), u.size(), 1e-5, 1e-6);
  }
}

int main() { return s2s::run_all(); }
