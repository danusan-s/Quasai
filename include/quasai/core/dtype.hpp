#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace quasai::core {

/**
 * @brief Supported data types for tensors.
 */
typedef enum {
  FLOAT32, ///< 32-bit floating point
  FLOAT64, ///< 64-bit floating point
  INT32,   ///< 32-bit signed integer
  INT64,   ///< 64-bit signed integer
} DType;

/**
 * @brief Dispatch a callable based on the tensor's data type.
 * @tparam F Type of the callable (must be invocable with <float>, <double>,
 * etc.)
 * @param dtype Data type to dispatch on.
 * @param f Callable to invoke with the appropriate type.
 * @return Result of calling f with the matching type.
 * @throws std::runtime_error if dtype is not supported.
 */
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

/**
 * @brief Traits class mapping a C++ type to its DType and properties.
 * @tparam T C++ type (float, double, int32_t, int64_t).
 */
template <typename T> struct DTypeTraits;

/// @brief Traits for float (FLOAT32).
template <> struct DTypeTraits<float> {
  static constexpr DType dtype = DType::FLOAT32;     ///< Corresponding DType
  static constexpr std::size_t size = sizeof(float); ///< Size in bytes
  static constexpr bool is_float = true;             ///< True if floating point
  static constexpr bool is_integer = false;          ///< True if integer
  static constexpr bool is_signed = false;           ///< True if signed
  static constexpr const char *name = "float32";     ///< Human-readable name
};

/// @brief Traits for double (FLOAT64).
template <> struct DTypeTraits<double> {
  static constexpr DType dtype = DType::FLOAT64;      ///< Corresponding DType
  static constexpr std::size_t size = sizeof(double); ///< Size in bytes
  static constexpr bool is_float = true;         ///< True if floating point
  static constexpr bool is_integer = false;      ///< True if integer
  static constexpr bool is_signed = false;       ///< True if signed
  static constexpr const char *name = "float64"; ///< Human-readable name
};

/// @brief Traits for int32_t (INT32).
template <> struct DTypeTraits<int32_t> {
  static constexpr DType dtype = DType::INT32;         ///< Corresponding DType
  static constexpr std::size_t size = sizeof(int32_t); ///< Size in bytes
  static constexpr bool is_float = false;      ///< True if floating point
  static constexpr bool is_integer = true;     ///< True if integer
  static constexpr bool is_signed = true;      ///< True if signed
  static constexpr const char *name = "int32"; ///< Human-readable name
};

/// @brief Traits for int64_t (INT64).
template <> struct DTypeTraits<int64_t> {
  static constexpr DType dtype = DType::INT64;         ///< Corresponding DType
  static constexpr std::size_t size = sizeof(int64_t); ///< Size in bytes
  static constexpr bool is_float = false;      ///< True if floating point
  static constexpr bool is_integer = true;     ///< True if integer
  static constexpr bool is_signed = true;      ///< True if signed
  static constexpr const char *name = "int64"; ///< Human-readable name
};

/**
 * @brief Get the size in bytes of a DType.
 * @param dtype Data type to query.
 * @return Size in bytes.
 * @throws std::runtime_error if dtype is not supported.
 */
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

/**
 * @brief Check if a DType is a floating-point type.
 * @param dtype Data type to query.
 * @return true if FLOAT32 or FLOAT64.
 * @throws std::runtime_error if dtype is not supported.
 */
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
