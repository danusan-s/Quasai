#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(Matmul, Float32) {
  std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data_b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  quasai::Shape shape_a{2, 3};
  quasai::Shape shape_b{3, 2};
  quasai::Tensor tensor_a =
      quasai::Tensor::from_data(data_a.data(), shape_a, quasai::DType::FLOAT32);
  quasai::Tensor tensor_b =
      quasai::Tensor::from_data(data_b.data(), shape_b, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::matmul(tensor_a, tensor_b);
  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 58.0f);
  EXPECT_FLOAT_EQ(result_data[1], 64.0f);
  EXPECT_FLOAT_EQ(result_data[2], 139.0f);
  EXPECT_FLOAT_EQ(result_data[3], 154.0f);
}
