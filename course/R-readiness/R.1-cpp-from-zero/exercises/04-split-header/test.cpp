#include <s2s/check.hpp>
#include <stats.hpp>  // declarations only: the test learns the signatures, not the code

S2S_TEST(mean_basic) {
  const double a[4] = {1, 2, 3, 4};
  CHECK_NEAR(mean(a, 4), 2.5, 1e-12, 0);
}

S2S_TEST(variance_basic) {
  const double a[4] = {2, 4, 4, 4};
  // mean 3.5, deviations -1.5 0.5 0.5 0.5 -> squares 2.25 0.25 0.25 0.25 -> sum 3 -> /4
  CHECK_NEAR(variance(a, 4), 0.75, 1e-12, 0);
}

S2S_TEST(range_basic) {
  const double a[5] = {3, -2, 7, 0, 5};
  CHECK_NEAR(range(a, 5), 9.0, 0, 0);
}

S2S_TEST(empty_input_gives_zero) {
  CHECK_NEAR(mean(nullptr, 0), 0.0, 0, 0);
  CHECK_NEAR(variance(nullptr, 0), 0.0, 0, 0);
  CHECK_NEAR(range(nullptr, 0), 0.0, 0, 0);
}

S2S_TEST(inline_helper_is_usable_from_this_file_too) {
  CHECK_NEAR(square(1.5), 2.25, 0, 0);
}

S2S_TEST_MAIN()
