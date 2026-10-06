#include <algorithm>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static long long kadane(const std::vector<int>& a) {
  long long best = a[0], cur = a[0];
  for (size_t i = 1; i < a.size(); ++i) { cur = std::max<long long>(a[i], cur + a[i]); best = std::max(best, cur); }
  return best;
}

static void check(std::vector<int> a) {
  s2s::DeviceBuffer<int> d(a);
  s2s::DeviceBuffer<long long> out(1);
  solve(d.get(), out.get(), int(a.size()));
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(out.download()[0], kadane(a));
}

S2S_TEST(small) {
  check({5});
  check({-3});
  check({-3, -1, -2});                  // all negative: the best single element
  check({2, -1, 2, -10, 3, 4, -1});
}

S2S_TEST(random) {
  for (int n : {1000, 65537, 1 << 20}) check(s2s::random_vec<int>(size_t(n), -100, 95, unsigned(n)));
}

S2S_TEST(combine_is_associative) {
  auto v = s2s::random_vec<int>(64, -50, 50, 3);
  Seg l = leaf(v[0]);
  for (int i = 1; i < 32; ++i) l = combine(l, leaf(v[size_t(i)]));
  Seg r = leaf(v[32]);
  for (int i = 33; i < 64; ++i) r = combine(r, leaf(v[size_t(i)]));
  CHECK_EQ(combine(l, r).best, kadane(v));
}

int main() { return s2s::run_all(); }
