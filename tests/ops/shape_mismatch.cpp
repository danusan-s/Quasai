#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(ShapeMismatchAdd, Float32) {
  quasai::core::Tensor tensor_a = quasai::core::Tensor::zeros(quasai::core::Shape{2, 3});
  quasai::core::Tensor tensor_b = quasai::core::Tensor::zeros(quasai::core::Shape{3, 2});

  EXPECT_THROW(quasai::ops::add(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchSub, Float32) {
  quasai::core::Tensor tensor_a = quasai::core::Tensor::zeros(quasai::core::Shape{2, 3});
  quasai::core::Tensor tensor_b = quasai::core::Tensor::zeros(quasai::core::Shape{3, 2});

  EXPECT_THROW(quasai::ops::sub(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchMul, Float32) {
  quasai::core::Tensor tensor_a = quasai::core::Tensor::zeros(quasai::core::Shape{2, 3});
  quasai::core::Tensor tensor_b = quasai::core::Tensor::zeros(quasai::core::Shape{3, 2});

  EXPECT_THROW(quasai::ops::mul(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchDiv, Float32) {
  quasai::core::Tensor tensor_a = quasai::core::Tensor::zeros(quasai::core::Shape{2, 3});
  quasai::core::Tensor tensor_b = quasai::core::Tensor::zeros(quasai::core::Shape{3, 2});

  EXPECT_THROW(quasai::ops::div(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchMatMul, Float32) {
  quasai::core::Tensor tensor_a = quasai::core::Tensor::zeros(quasai::core::Shape{2, 3});
  quasai::core::Tensor tensor_b = quasai::core::Tensor::zeros(quasai::core::Shape{4, 2});

  EXPECT_THROW(quasai::ops::matmul(tensor_a, tensor_b), std::runtime_error);
}
