#include <s2s/check.hpp>
#include <impl.hpp>

S2S_TEST(ramp_of_four) {
  int* p = make_ramp(4);
  CHECK_EQ(p[0], 0);
  CHECK_EQ(p[1], 10);
  CHECK_EQ(p[2], 20);
  CHECK_EQ(p[3], 30);
  delete[] p;
}

S2S_TEST(ramp_of_one) {
  int* p = make_ramp(1);
  CHECK_EQ(p[0], 0);
  delete[] p;
}

S2S_TEST_MAIN()
