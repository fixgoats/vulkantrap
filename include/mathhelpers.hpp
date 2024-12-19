#pragma once
#include "betterexc.h"
#include "typedefs.hpp"
#include "vkhelpers.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

using std::bit_cast;

template <typename T>
constexpr uint32_t euclid_mod(T a, uint32_t b) {
  assert(b != 0);
  return a % b;
}

constexpr float S1(cvec2 x) {
  return 2 * (x.x.real() * x.y.real() + x.x.imag() * x.y.imag());
}
constexpr float S2(cvec2 x) {
  return 2 * (x.x.real() * x.y.imag() - x.x.imag() * x.y.real());
}
constexpr float S3(cvec2 x) { return std::norm(x.x) - std::norm(x.y); }

template <class T, uint32_t C, uint32_t R>
struct small_mat {
  std::array<T, R * C> buffer;

  // it seems like in at least some cases memcpying an array of arrays with the
  // total size does give you a flattened array, but I don't think this is
  // something I can count on.
  small_mat(const T (&ll)[R][C]) {
    for (size_t j = 0; j < R; j++) {
      for (size_t i = 0; i < C; i++) {
        buffer[j * C + i] = ll[j][i];
      }
    }
  }

  constexpr uint32_t X() { return C; }
  constexpr uint32_t Y() { return R; }
};

// For when you need a more flexible matrix.
template <class T>
struct mat {
  static_assert(std::is_same_v<T, float> or std::is_same_v<T, c32> or
                    std::is_same_v<T, double> or std::is_same_v<T, c64>,
                "Type must be one of: float, c32, double or c64");
  std::array<uint32_t, 2> dims;
  std::vector<T> buffer;

  template <uint32_t C, uint32_t R>
  mat(const T (&ll)[C][R]) {
    dims = {C, R};
    buffer.resize(R * C);
    for (size_t j = 0; j < R; j++) {
      for (size_t i = 0; i < C; i++) {
        buffer[j * C + i] = ll[j][i];
      }
    }
  }
  mat() {
    dims = {0, 0};
    buffer = {};
  }
  mat(uint32_t c, uint32_t r) {
    dims = {c, r};
    buffer.resize(c * r);
  }

  constexpr uint32_t X() { return dims[0]; }
  constexpr uint32_t Y() { return dims[1]; }
  constexpr T* data() { return buffer.data(); }
  constexpr size_t size() { return buffer.size(); }

  void savetxt(std::string fname) {
    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
      throw runtime_exc{"Couldn't open file: {}", fname};
    }

    file.write(reinterpret_cast<const char*>(dims.data()), 8);
    file.write(reinterpret_cast<const char*>(buffer.data()),
               buffer.size() * sizeof(T));
    file.close();
  }

  mat(const std::vector<char>& buf) {
    memcpy(dims.data(), buf.data(), 8);
    buffer.resize(dims[0] * dims[1]);
    memcpy(buffer.data(), buf.data() + 8, buf.size() - 8);
  }

  mat(const std::vector<char>&& buf) {
    memcpy(dims.data(), buf.data(), 8);
    buffer.resize(dims[0] * dims[1]);
    memcpy(buffer.data(), buf.data() + 8, buf.size() - 8);
  }
};

template <class T, uint32_t S, uint32_t C, uint32_t R>
struct small_arr3 {
  // small_* is just the array, dimensions are compile time-constants, hence
  // no specific serialization, just reading/writing the flat data.
  std::array<T, S * C * R> data;

  small_arr3(const T (&ll)[S][R][C]) {
    for (size_t k = 0; k < S; k++) {
      for (size_t j = 0; j < R; j++) {
        for (size_t i = 0; i < C; i++) {
          data[(k * R + j) * C + i] = ll[k][j][i];
        }
      }
    }
  }
  constexpr uint32_t X() { return R; }
  constexpr uint32_t Y() { return C; }
  constexpr uint32_t Z() { return S; }
};

template <class T>
struct arr3 {
  std::array<uint32_t, 3> dims;
  std::vector<T> data{};

  template <uint32_t C, uint32_t R, uint32_t S>
  constexpr arr3(const T (&ll)[S][R][C]) {
    /*std::cout << "slices: " << S << '\n';
    std::cout << "rows: " << R << '\n';
    std::cout << "columns: " << C << '\n';*/
    data.resize(S * C * R);
    for (size_t k = 0; k < S; k++) {
      for (size_t j = 0; j < R; j++) {
        for (size_t i = 0; i < C; i++) {
          data.push_back(ll[k][j][i]);
        }
      }
    }
  }

  constexpr arr3(uint32_t rows, uint32_t cols, uint32_t slices) {
    dims = {rows, cols, slices};
    std::vector<T> data(slices * cols * rows);
  }

  constexpr uint32_t X() { return dims[0]; }
  constexpr uint32_t Y() { return dims[1]; }
  constexpr uint32_t Z() { return dims[2]; }

  void save(std::string fname) {
    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
      throw runtime_exc{"Couldn't open file: {}", fname};
    }

    file.write(reinterpret_cast<const char*>(dims.data()), 12);
    file.write(reinterpret_cast<const char*>(data.data()),
               data.size() * sizeof(T));
    file.close();
  }

  // It may be in some way more performant and flexible to take a bare pointer
  // rather than a vector reference, but a reference is guaranteed to be
  // non-null and also allows capturing temporals:
  // https://stackoverflow.com/a/52255382/6461823
  arr3(const std::vector<char>& buf) {
    memcpy(dims.data(), buf.data(), 12);
    data.resize(dims[0] * dims[1] * dims[2]);
    memcpy(data.data(), buf.data() + 12, buf.size() - 12);
  }

  arr3(const std::vector<char>&& buf) {
    memcpy(dims.data(), buf.data(), 12);
    data.resize(dims[0] * dims[1] * dims[2]);
    memcpy(data.data(), buf.data() + 12, buf.size() - 12);
  }
};

template <typename T>
constexpr T square(T x) {
  return x * x;
}
constexpr uint32_t fftshiftidx(uint32_t i, uint32_t n) {
  return euclid_mod(i + (n + 1) / 2, n);
}

template <typename T>
void leftRotate(std::vector<T>& arr, uint32_t d) {
  auto n = arr.size();
  d = d % n; // To handle case when d >= n

  // Reverse the first d elements
  std::reverse(arr.begin(), arr.begin() + d);

  // Reverse the remaining elements
  std::reverse(arr.begin() + d, arr.end());

  // Reverse the whole array
  std::reverse(arr.begin(), arr.end());
}

template <typename T>
void fftshift(std::vector<T>& arr) {
  auto n = arr.size();
  uint32_t d = (n + 1) % 2;
  leftRotate(arr, d);
}

constexpr float pumpProfile(float x, float y, float L, float r, float beta) {
  return square(square(L)) /
         (square(x * x + beta * y * y - r * r) + square(square(L)));
}

template <typename T>
uint8_t mapToColor(T v, T min, T max) {
  return static_cast<uint8_t>(256 * (v - min) / (max - min));
}

template <typename InputIt, typename OutputIt>
void colorMap(InputIt it, InputIt end, OutputIt out) {
  const auto max = *std::max_element(it, end);
  const auto min = *std::min_element(it, end);
  std::transform(it, end, out, [&](auto x) { return mapToColor(x, min, max); });
}

template <typename InputIt>
std::vector<uint8_t> colorMapVec(InputIt it, InputIt end) {
  std::vector<uint8_t> out(end - it);
  colorMap(it, end, out.begin());
  return out;
}
