#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(TensorOpsTest, Sum) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::sum(tensor);

  const auto impl = result.get_impl();
  EXPECT_EQ(impl.shape.dimensions(), 0); // Scalar result
  EXPECT_EQ(impl.dtype, quasai::DType::FLOAT32);
  EXPECT_EQ(impl.device.type, quasai::CPU);

  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 10.0f); // Sum of all elements
}

TEST(TensorOpsTest, Mean) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::Shape shape{2, 2};
  quasai::Tensor tensor =
      quasai::Tensor::from_data(data.data(), shape, quasai::DType::FLOAT32);

  quasai::Tensor result = quasai::mean(tensor);

  const auto impl = result.get_impl();
  EXPECT_EQ(impl.shape.dimensions(), 0); // Scalar result
  EXPECT_EQ(impl.dtype, quasai::DType::FLOAT32);
  EXPECT_EQ(impl.device.type, quasai::CPU);

  float *result_data = result.data<float>();
  EXPECT_FLOAT_EQ(result_data[0], 2.5f); // Mean of all elements
}
