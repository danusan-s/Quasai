#pragma once
/**
 * @file types.h
 * @brief Core type definitions for Quasai framework
 * 
 * This file defines fundamental types used throughout the framework:
 * - Type aliases (index_t, size_t, real_t)
 * - Enum types (DataType, DeviceType)
 * - Constants (EPSILON, PI)
 * - Type traits
 */

#include <cstddef>
#include <cstdint>

namespace quasai {

using index_t = std::size_t;       ///< Index type for tensor dimensions
using size_t = std::size_t;        ///< Size type
using real_t = float;              ///< Default floating point precision

enum class DataType {
    FLOAT32 = 0,
    FLOAT64 = 1,
    INT32 = 2,
    INT64 = 3,
    BOOL = 4
};

enum class DeviceType {
    CPU = 0,
    GPU = 1
};

constexpr index_t MAX_DIMENSIONS = 8;
constexpr real_t EPSILON = 1e-7f;
constexpr real_t PI = 3.14159265358979f;

} // namespace quasai
