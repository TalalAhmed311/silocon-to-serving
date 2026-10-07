#include <s2s/check.hpp>
#include "impl.hpp"

S2S_TEST(push_through_growth) {
  TokenRing r(2);
  for (int i = 0; i < 100; ++i) {
    int32_t* p = r.push(i);
    CHECK_EQ(*p, i);
  }
  CHECK_EQ(r.size(), 100);
  for (int i = 0; i < 100; ++i) CHECK_EQ(r.at(i), i);
}

S2S_TEST(clear_stays_in_bounds) {
  TokenRing r(4);
  for (int i = 0; i < 4; ++i) r.push(i + 1);
  r.clear();
  CHECK_EQ(r.size(), 0);
}

S2S_TEST(fingerprint_defined_and_order_sensitive) {
  TokenRing a(8), b(8);
  for (int i = 0; i < 8; ++i) { a.push(i + 1); b.push(8 - i); }
  CHECK(a.fingerprint(8) != b.fingerprint(8));
  CHECK_EQ(a.fingerprint(8), a.fingerprint(8));
}

S2S_TEST_MAIN()
