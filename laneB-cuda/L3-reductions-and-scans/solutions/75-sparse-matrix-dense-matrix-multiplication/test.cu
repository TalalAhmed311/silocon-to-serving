#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int M, int N, int P, int max_per_row) {
  std::vector<int> rp{0}, ci;
  std::vector<float> v;
  auto counts = s2s::random_vec<int>(size_t(M), 0, max_per_row, unsigned(M));
  for (int r = 0; r < M; ++r) {
    auto c = s2s::random_vec<int>(size_t(counts[size_t(r)]), 0, N - 1, unsigned(r + 3));
    auto x = s2s::random_vec<float>(c.size(), -1, 1, unsigned(r + 5));
    ci.insert(ci.end(), c.begin(), c.end());
    v.insert(v.end(), x.begin(), x.end());
    rp.push_back(int(ci.size()));
  }
  auto B = s2s::random_vec<float>(size_t(N) * P, -1, 1, 11);
  std::vector<float> want(size_t(M) * P, 0.f);
  for (int r = 0; r < M; ++r)
    for (int c = 0; c < P; ++c) {
      double acc = 0;
      for (int k = rp[size_t(r)]; k < rp[size_t(r) + 1]; ++k) acc += double(v[size_t(k)]) * B[size_t(ci[size_t(k)]) * P + c];
      want[size_t(r) * P + c] = float(acc);
    }
  if (ci.empty()) { ci.push_back(0); v.push_back(0); }
  s2s::DeviceBuffer<int> drp(rp), dci(ci);
  s2s::DeviceBuffer<float> dv(v), dB(B), dC(size_t(M) * P);
  solve(drp.get(), dci.get(), dv.get(), dB.get(), dC.get(), M, N, P);
  CUDA_CHECK_LAUNCH();
  auto got = dC.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-5);
}

S2S_TEST(shapes) {
  check(1, 1, 1, 1);
  check(50, 40, 33, 5);
  check(300, 200, 130, 300);     // rows longer than one CHUNK, P not a multiple of the block
  check(64, 64, 256, 0);         // all-empty rows → zeros
}

int main() { return s2s::run_all(); }
