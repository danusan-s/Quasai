#pragma once

#include "quasai/ops/core_ops.hpp"

namespace quasai::ops {

template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

template <Scalar T> core::Tensor add(const core::Tensor &a, T &scalar) {
  return add(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> core::Tensor add(T &scalar, const core::Tensor &a) {
  return add(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> core::Tensor sub(const core::Tensor &a, T &scalar) {
  return sub(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> core::Tensor sub(T &scalar, const core::Tensor &a) {
  return sub(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> core::Tensor mul(const core::Tensor &a, T &scalar) {
  return mul(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> core::Tensor mul(T &scalar, const core::Tensor &a) {
  return mul(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> core::Tensor div(const core::Tensor &a, T &scalar) {
  return div(a, core::Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> core::Tensor div(T &scalar, const core::Tensor &a) {
  return div(core::Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

} // namespace quasai::ops
