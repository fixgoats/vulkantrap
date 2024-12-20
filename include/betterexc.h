#pragma once
#include <format>
#include <stdexcept>

class runtime_exc : public std::runtime_error {
public:
  template <class... Args>
  runtime_exc(std::format_string<Args...> what_arg_fmt, Args&&... args)
      : runtime_error{std::format(what_arg_fmt, args...)} {}
};
