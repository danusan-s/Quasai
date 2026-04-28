#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace quasai::core {

typedef enum {
  FLOAT32,
  FLOAT64,
  INT32,
  INT64,
} DType;

// C++20 dtype dispatch helper using generic lambdas
template <typename F> constexpr auto dispatch_by_dtype(DType dtype, F &&f) {
  switch (dtype) {
    case DType::FLOAT32:
      return std::forward<F>(f).template operator()<float>();
    case DType::FLOAT64:
      return std::forward<F>(f).template operator()<double>();
    case DType::INT32:
      return std::forward<F>(f).template operator()<int32_t>();
    case DType::INT64:
      return std::forward<F>(f).template operator()<int64_t>();
    default:
      throw std::runtime_error("Unsupported data type for dispatch");
  }
}

template <typename T> struct DTypeTraits;

template <> struct DTypeTraits<float> {
  static constexpr DType dtype = DType::FLOAT32;
  static constexpr std::size_t size = sizeof(float);
  static constexpr bool is_float = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = false;
  static constexpr const char *name = "float32";
};

template <> struct DTypeTraits<double> {
  static constexpr DType dtype = DType::FLOAT64;
  static constexpr std::size_t size = sizeof(double);
  static constexpr bool is_float = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = false;
  static constexpr const char *name = "float64";
};

template <> struct DTypeTraits<int32_t> {
  static constexpr DType dtype = DType::INT32;
  static constexpr std::size_t size = sizeof(int32_t);
  static constexpr bool is_float = false;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
  static constexpr const char *name = "int32";
};

template <> struct DTypeTraits<int64_t> {
  static constexpr DType dtype = DType::INT64;
  static constexpr std::size_t size = sizeof(int64_t);
  static constexpr bool is_float = false;
  static constexpr bool is_integer = true;
  static constexpr bool is_signed = true;
  static constexpr const char *name = "int64";
};

inline std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::FLOAT32:
      return DTypeTraits<float>::size;
    case DType::FLOAT64:
      return DTypeTraits<double>::size;
    case DType::INT32:
      return DTypeTraits<int32_t>::size;
    case DType::INT64:
      return DTypeTraits<int64_t>::size;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

inline bool is_floating(DType dtype) {
  switch (dtype) {
    case DType::FLOAT32:
    case DType::FLOAT64:
      return true;
    case DType::INT32:
    case DType::INT64:
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

} // namespace quasai::core
