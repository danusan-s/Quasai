// src/core/error.h
// Header for error handling

#pragma once
/**
 * @file error.h
 * @brief Error handling and exception definitions
 *
 * Define custom exception classes and error codes for the framework.
 * Implement proper error propagation and debugging utilities.
 */

#include <stdexcept>
#include <string>

namespace quasai {

// Base exception class
class Exception : public std::runtime_error {
public:
  explicit Exception(const std::string &message) : std::runtime_error(message) {
  }
  explicit Exception(const std::string &message, int error_code)
      : std::runtime_error(message), error_code_(error_code) {
  }
  int error_code_ = 0;
};

// Exception for invalid parameters
class InvalidParameterException : public Exception {
public:
  explicit InvalidParameterException(const std::string &message)
      : Exception(message, 1001) {
  }
};

// Exception for dimension mismatch
class DimensionException : public Exception {
public:
  explicit DimensionException(const std::string &message)
      : Exception(message, 1002) {
  }
};

} // namespace quasai
