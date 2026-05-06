#pragma once

#include "quasai/ops/core_ops.hpp"

namespace quasai::ops {

/// @brief Concept for scalar types (arithmetic types).
template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

/// @brief Add a scalar to a tensor (scalar on right).
template <Scalar T> core::Tensor add(const core::Tensor &a, T &scalar) {
  return add(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

/// @brief Add a scalar to a tensor (scalar on left).
template <Scalar T> core::Tensor add(T &scalar, const core::Tensor &a) {
  return add(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

/// @brief Subtract a scalar from a tensor.
template <Scalar T> core::Tensor sub(const core::Tensor &a, T &scalar) {
  return sub(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

/// @brief Subtract a tensor from a scalar.
template <Scalar T> core::Tensor sub(T &scalar, const core::Tensor &a) {
  return sub(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

/// @brief Multiply a tensor by a scalar (scalar on right).
template <Scalar T> core::Tensor mul(const core::Tensor &a, T &scalar) {
  return mul(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

/// @brief Multiply a tensor by a scalar (scalar on left).
template <Scalar T> core::Tensor mul(T &scalar, const core::Tensor &a) {
  return mul(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

/// @brief Divide a tensor by a scalar.
template <Scalar T> core::Tensor div(const core::Tensor &a, T &scalar) {
  return div(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

/// @brief Divide a scalar by a tensor.
template <Scalar T> core::Tensor div(T &scalar, const core::Tensor &a) {
  return div(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

} // namespace quasai::ops
