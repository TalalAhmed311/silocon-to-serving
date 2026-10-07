#include <d4/elementwise.cuh>
#include <d4/transpose.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

S2S_TEST(vadd_both) {
  for (long long n : {1LL, 5LL, 1LL << 20}) {
    auto a = s2s::random_vec<float>(size_t(n), -1, 1, 1), b = s2s::random_vec<float>(size_t(n), -1, 1, 2);
    std::vector<float> want(a.size());
    for (size_t i = 0; i < a.size(); ++i) want[i] = a[i] + b[i];
    for (bool v4 : {false, true}) {
      s2s::DeviceBuffer<float> da(a), db(b), dc(a.size());
      d4::vadd(da.get(), db.get(), dc.get(), n, v4);
      CUDA_CHECK_LAUNCH();
      auto got = dc.download();
      CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
    }
  }
}

S2S_TEST(transpose_rungs) {
  for (auto [R, C] : {std::pair{33, 65}, std::pair{1024, 512}, std::pair{1, 1}}) {
    auto in = s2s::random_vec<float>(size_t(R) * C);
    std::vector<float> want(in.size());
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) want[size_t(c) * R + r] = in[size_t(r) * C + c];
    for (int rung = 1; rung <= 3; ++rung) {
      s2s::DeviceBuffer<float> din(in), dout(in.size());
      d4::transpose(din.get(), dout.get(), R, C, rung);
      CUDA_CHECK_LAUNCH();
      auto got = dout.download();
      CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
    }
  }
}

S2S_TEST_MAIN()
