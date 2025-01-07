#pragma once
#include <complex>
#include <cstdint>

typedef std::complex<float> c32;
typedef std::complex<double> c64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef float f32;
typedef double f64;

template <class T>
constexpr auto numfmt(T x) {
  if constexpr (std::is_same_v<T, c32> or std::is_same_v<T, c64>) {
    return std::format("({}+{}j)", x.real(), x.imag());
  } else {
    return std::format("{}", x);
  }
}

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
  friend std::ostream& operator<<(std::ostream& stream, const cvec2& v) {
    stream << numfmt(v.x) << ' ' << numfmt(v.y);
    return stream;
  }
};
