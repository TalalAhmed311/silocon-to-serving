#include <cmath>
#include <random>
#include <vector>

#include <s2s/check.hpp>
#include <ops.hpp>

static std::string dir;

// logits for probabilities {0.5, 0.3, 0.15, 0.05} at temperature 1
static std::vector<float> L = {std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};

S2S_TEST(greedy_when_temperature_zero) { CHECK_EQ(ops::sample_top_p(L.data(), 4, 0.0f, 0.9f, 0.99f), 0); }

S2S_TEST(top_p_small_keeps_only_argmax) {
  for (float u : {0.0f, 0.5f, 0.999f}) CHECK_EQ(ops::sample_top_p(L.data(), 4, 1.0f, 0.4f, u), 0);
}

S2S_TEST(chi_square_matches_truncated_distribution) {
  // top_p = 0.79 keeps {0.5, 0.3} (cumulative 0.8 >= 0.79): renormalized to {0.625, 0.375}; tokens 2, 3 never appear.
  // (Not 0.8 exactly: float rounding can make 0.5 + 0.3 land a hair below 0.8 and keep a third token.)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  const int N = 100000;
  int counts[4] = {0, 0, 0, 0};
  for (int i = 0; i < N; ++i) counts[ops::sample_top_p(L.data(), 4, 1.0f, 0.79f, uni(rng))]++;
  CHECK_EQ(counts[2] + counts[3], 0);
  const double e0 = 0.625 * N, e1 = 0.375 * N;
  const double chi2 = (counts[0] - e0) * (counts[0] - e0) / e0 + (counts[1] - e1) * (counts[1] - e1) / e1;
  CHECK(chi2 < 10.83);  // 1 degree of freedom, p = 0.001 critical value
}

S2S_TEST(top_p_one_keeps_everything) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  int seen3 = 0;
  for (int i = 0; i < 20000; ++i) seen3 += ops::sample_top_p(L.data(), 4, 1.0f, 1.0f, uni(rng)) == 3;
  CHECK(seen3 > 700 && seen3 < 1300);  // expect 5% = 1000
}

S2S_TEST(temperature_sharpens) {
  // T = 0.25 makes token 0 dominate (0.5^4 / Σ p_i^4 ≈ 0.88)
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  int zero = 0;
  for (int i = 0; i < 10000; ++i) zero += ops::sample_top_p(L.data(), 4, 0.25f, 1.0f, uni(rng)) == 0;
  CHECK(zero > 8500);
}

int main(int argc, char** argv) { dir = argc > 1 ? argv[1] : ""; return s2s::run_all(); }
