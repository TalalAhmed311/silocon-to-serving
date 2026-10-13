#pragma once
#include <cmath>

struct Vec2 {
  float x;
  float y;

  Vec2() : x(0), y(0) {}
  Vec2(float x_, float y_) : x(x_), y(y_) {}

  float length() const { return std::sqrt(x * x + y * y); }

  Vec2 operator+(const Vec2& o) const { return Vec2(x + o.x, y + o.y); }

  float dot(const Vec2& o) const { return x * o.x + y * o.y; }
};
