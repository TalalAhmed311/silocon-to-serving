// Exercise 3 starter. TODO: write the constructor bodies and methods of Vec2.
#pragma once
#include <cmath>

struct Vec2 {
  float x;
  float y;

  // Default: (0, 0). TODO: use a member initializer list  Vec2() : x(0), y(0) {}
  Vec2() : x(1), y(1) {}

  // TODO: set x and y from the arguments, using a member initializer list.
  Vec2(float x_, float y_) : x(1), y(1) { (void)x_; (void)y_; }

  // sqrt(x*x + y*y). The `const` after the parameter list promises not to modify *this,
  // which is what lets you call it on a `const Vec2`.
  float length() const {
    // TODO: use std::sqrt
    return 0.0f;
  }

  // Component-wise sum. Returns a new Vec2.
  Vec2 operator+(const Vec2& o) const {
    // TODO
    (void)o;
    return Vec2();
  }

  // Dot product: x*o.x + y*o.y.
  float dot(const Vec2& o) const {
    // TODO
    (void)o;
    return 0.0f;
  }
};
