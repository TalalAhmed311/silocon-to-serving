#include <s2s/check.hpp>
#include <impl.hpp>  // <> so the include path (starter or solutions) decides, not this file's folder

S2S_TEST(default_is_zero) {
  Vec2 v;
  CHECK_NEAR(v.x, 0.0, 0, 0);
  CHECK_NEAR(v.y, 0.0, 0, 0);
}

S2S_TEST(constructor_sets_fields) {
  Vec2 v(3.0f, -4.5f);
  CHECK_NEAR(v.x, 3.0, 0, 0);
  CHECK_NEAR(v.y, -4.5, 0, 0);
}

S2S_TEST(length_on_a_const_object) {
  const Vec2 v(3.0f, 4.0f);  // a const object: only const methods may be called
  CHECK_NEAR(v.length(), 5.0, 1e-6, 0);
}

S2S_TEST(plus_makes_a_new_vector) {
  const Vec2 a(1.0f, 2.0f), b(10.0f, 20.0f);
  Vec2 c = a + b;
  CHECK_NEAR(c.x, 11.0, 0, 0);
  CHECK_NEAR(c.y, 22.0, 0, 0);
  CHECK_NEAR(a.x, 1.0, 0, 0);  // operands unchanged
  CHECK_NEAR(b.y, 20.0, 0, 0);
}

S2S_TEST(dot_product) {
  const Vec2 a(1.0f, 2.0f), b(3.0f, 4.0f);
  CHECK_NEAR(a.dot(b), 11.0, 0, 0);
}

S2S_TEST(sizeof_is_two_floats) {
  static_assert(sizeof(Vec2) == 8, "Vec2 must hold exactly two floats");
  CHECK_EQ(int(sizeof(Vec2)), 8);
}

S2S_TEST_MAIN()
