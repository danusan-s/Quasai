#pragma once

#include "quasai/ops/core_ops.hpp"

namespace quasai {

template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

template <Scalar T> Tensor add(const Tensor &a, T &scalar) {
  return add(a, Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> Tensor add(T &scalar, const Tensor &a) {
  return add(Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> Tensor sub(const Tensor &a, T &scalar) {
  return sub(a, Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> Tensor sub(T &scalar, const Tensor &a) {
  return sub(Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> Tensor mul(const Tensor &a, T &scalar) {
  return mul(a, Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> Tensor mul(T &scalar, const Tensor &a) {
  return mul(Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

template <Scalar T> Tensor div(const Tensor &a, T &scalar) {
  return div(a, Tensor::from_scalar(scalar, a.dtype(), a.device()));
}

template <Scalar T> Tensor div(T &scalar, const Tensor &a) {
  return div(Tensor::from_scalar(scalar, a.dtype(), a.device()), a);
}

} // namespace quasai
