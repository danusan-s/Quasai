#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(ShapeMismatchAdd, Float32) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::add(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchSub, Float32) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::sub(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchMul, Float32) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::mul(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchDiv, Float32) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{3, 2});

  EXPECT_THROW(quasai::div(tensor_a, tensor_b), std::runtime_error);
}

TEST(ShapeMismatchMatMul, Float32) {
  quasai::Tensor tensor_a = quasai::Tensor::zeros(quasai::Shape{2, 3});
  quasai::Tensor tensor_b = quasai::Tensor::zeros(quasai::Shape{4, 2});

  EXPECT_THROW(quasai::matmul(tensor_a, tensor_b), std::runtime_error);
}
