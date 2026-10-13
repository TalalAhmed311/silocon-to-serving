#include <limits>
#include <s2s/check.hpp>
#include <impl.hpp>  // <> so the include path (starter or solutions) decides, not this file's folder

S2S_TEST(sum_of_small_array) {
  const float a[5] = {1, 2, 3, 4, 5};
  CHECK_NEAR(array_sum(a, 5), 15.0, 0, 1e-6);
}

S2S_TEST(max_of_small_array) {
  const float a[5] = {3, -1, 9, 2, 9};
  CHECK_NEAR(array_max(a, 5), 9.0, 0, 0);
}

S2S_TEST(max_of_all_negative_values) {
  const float a[3] = {-5, -2, -8};
  CHECK_NEAR(array_max(a, 3), -2.0, 0, 0);
}

S2S_TEST(single_element) {
  const float a[1] = {42};
  CHECK_NEAR(array_sum(a, 1), 42.0, 0, 0);
  CHECK_NEAR(array_max(a, 1), 42.0, 0, 0);
}

S2S_TEST(empty_array_is_handled_and_not_read) {
  // nullptr with n == 0 is legal: the functions must not dereference it.
  CHECK_NEAR(array_sum(nullptr, 0), 0.0, 0, 0);
  CHECK(array_max(nullptr, 0) == std::numeric_limits<float>::lowest());
}

S2S_TEST(sub_range_via_pointer_offset) {
  const float a[6] = {100, 1, 2, 3, 4, 100};
  CHECK_NEAR(array_sum(a + 1, 4), 10.0, 0, 1e-6);
  CHECK_NEAR(array_max(a + 1, 4), 4.0, 0, 0);
}

S2S_TEST_MAIN()
