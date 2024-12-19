#pragma once
#include <complex>

typedef std::complex<float> c32;
typedef std::complex<double> c64;

struct cvec2 {
  c32 x;
  c32 y;

  constexpr cvec2 operator*(const cvec2 b) const { return {x * b.x, y * b.y}; }
  template <class T>
  constexpr cvec2 operator*(const T b) const {
    return {b * x, b * y};
  }
  constexpr cvec2 operator+(const cvec2 b) const { return {x + b.x, y + b.y}; }
  template <class T>
  friend constexpr cvec2 operator*(const T a, const cvec2 b) {
    return {a * b.x, a * b.y};
  }
};
