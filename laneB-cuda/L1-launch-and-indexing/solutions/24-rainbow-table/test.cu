#include <cstdint>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(matches_cpu) {
  for (int R : {1, 3, 100}) {
    const int n = 10007;
    auto in = s2s::random_vec<int32_t>(size_t(n), -1000000, 1000000, unsigned(R));
    s2s::DeviceBuffer<int32_t> din(in);
    s2s::DeviceBuffer<uint32_t> dout(size_t(n));
    solve(din.get(), dout.get(), n, R);
    CUDA_CHECK_LAUNCH();
    auto out = dout.download();
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      uint32_t v = uint32_t(in[size_t(i)]);
      for (int r = 0; r < R; ++r) v = hash_round(v);
      ok &= out[size_t(i)] == v;
    }
    CHECK(ok);
  }
}

int main() { return s2s::run_all(); }
