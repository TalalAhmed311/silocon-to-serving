# Exercise 3: A small `Vec2` class (easy)

`impl.hpp` has the shape of a `struct Vec2` with two `float` fields. The constructors and methods are stubs.

**Task:** make them work.

- Both constructors use a **member initializer list**: `Vec2(float x_, float y_) : x(x_), y(y_) {}`. The default constructor sets both to 0.
- `length()` returns `sqrt(x*x + y*y)`.
- `operator+` returns a new `Vec2`. It must not change either operand.
- `dot()` returns `x*o.x + y*o.y`.

Every method already carries `const` after its parameter list. Keep it. The test calls `length()` on a `const Vec2`, and the compiler only allows that for `const` methods.

**Hints**

1. `operator+` is a normal method with a funny name. `a + b` means `a.operator+(b)`.
2. Take the other vector as `const Vec2&` so no copy is made.
3. Do not add any fields. The test checks that `sizeof(Vec2) == 8`.

**Test:** `ctest -R 03-vec2-class --output-on-failure`.
