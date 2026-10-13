#include <s2s/check.hpp>
#include <impl.hpp>  // <> so the include path (starter or solutions) decides, not this file's folder

S2S_TEST(pointer_swap_exchanges_values) {
  int a = 1, b = 2;
  swap_by_pointer(&a, &b);
  CHECK_EQ(a, 2);
  CHECK_EQ(b, 1);
}

S2S_TEST(reference_swap_exchanges_values) {
  int a = -7, b = 40;
  swap_by_ref(a, b);
  CHECK_EQ(a, 40);
  CHECK_EQ(b, -7);
}

S2S_TEST(swap_inside_an_array) {
  int v[3] = {10, 20, 30};
  swap_by_pointer(&v[0], &v[2]);
  CHECK_EQ(v[0], 30);
  CHECK_EQ(v[1], 20);
  CHECK_EQ(v[2], 10);
  swap_by_ref(v[0], v[1]);
  CHECK_EQ(v[0], 20);
  CHECK_EQ(v[1], 30);
}

S2S_TEST(swapping_an_int_with_itself_changes_nothing) {
  int a = 5;
  swap_by_pointer(&a, &a);
  CHECK_EQ(a, 5);
  swap_by_ref(a, a);
  CHECK_EQ(a, 5);
}

S2S_TEST_MAIN()
