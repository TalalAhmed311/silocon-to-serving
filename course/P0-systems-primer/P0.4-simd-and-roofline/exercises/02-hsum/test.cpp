#include <random>

#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(random_vectors) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-10, 10);
  for (int k = 0; k < 1000; ++k) {
    float x[8];
    double want = 0;
    for (int i = 0; i < 8; ++i) { x[i] = d(rng); }
#if defined(__AVX2__)
    for (int i = 0; i < 8; ++i) want += x[i];
    CHECK_NEAR(hsum8(_mm256_loadu_ps(x)), want, 1e-5, 1e-5);
#elif defined(__ARM_NEON)
    for (int i = 0; i < 4; ++i) want += x[i];
    CHECK_NEAR(hsum4(vld1q_f32(x)), want, 1e-5, 1e-5);
#endif
  }
}

S2S_TEST_MAIN()
