#include <s2s/check.hpp>
#include <impl.hpp>

// Overwrites a big patch of the stack, so a dangling pointer to an old frame reads garbage.
static int scribble_on_the_stack(int seed) {
  volatile int junk[256];
  for (int i = 0; i < 256; ++i) junk[i] = seed + i;
  return junk[seed & 255];
}

S2S_TEST(values_survive_after_return) {
  auto s = first_squares(5);
  scribble_on_the_stack(1234);
  CHECK_EQ(s[0], 1);
  CHECK_EQ(s[1], 4);
  CHECK_EQ(s[2], 9);
  CHECK_EQ(s[3], 16);
  CHECK_EQ(s[4], 25);
}

S2S_TEST(two_results_do_not_share_storage) {
  auto a = first_squares(4);
  auto b = first_squares(6);  // a "fix" with a static array would make b overwrite a
  scribble_on_the_stack(99);
  CHECK_EQ(a[3], 16);
  CHECK_EQ(b[3], 16);
  CHECK_EQ(b[5], 36);
}

S2S_TEST_MAIN()
