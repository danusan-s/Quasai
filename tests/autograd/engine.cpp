#include "quasai/autograd/engine.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(AutoGradEngineTest, BackwardNoGrad) {
  // Create a tensor that does not require grad
  quasai::core::Tensor tensor =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});

  // Call backward and ensure it does not throw
  EXPECT_NO_THROW(tensor.backward());
}

TEST(AutoGradEngineTest, BackwardSimpleGraph) {
  // Create tensors that require grad
  quasai::core::Tensor a =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  a.requires_grad(true);
  quasai::core::Tensor b =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  b.requires_grad(true);

  // Create a simple computation graph: c = a + b
  quasai::core::Tensor c = quasai::ops::add(a, b);

  // Call backward on c
  EXPECT_NO_THROW(c.backward());

  // Check that gradients for a and b are created by engine the first time
  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a->grad_.buffer());
  EXPECT_TRUE(meta_b->grad_.buffer());

  float *grad_a = meta_a->grad_.data<float>();
  float *grad_b = meta_b->grad_.data<float>();

  // Since c = a + b, the gradient for both a and b should be ones
  for (size_t i = 0; i < quasai::core::total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 1.0f);
    EXPECT_FLOAT_EQ(grad_b[i], 1.0f);
  }
}

TEST(AutoGradEngineTest, BackwardAccumulateGrad) {
  // Create tensors that require grad
  quasai::core::Tensor a =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  a.requires_grad(true);
  quasai::core::Tensor b =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  b.requires_grad(true);

  // Create a computation graph: c = a + b, d = a + b
  quasai::core::Tensor c = quasai::ops::add(a, b);
  quasai::core::Tensor d = quasai::ops::add(a, b);

  // Call backward on c and d to test gradient accumulation
  EXPECT_NO_THROW(c.backward());
  EXPECT_NO_THROW(d.backward());

  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a);
  EXPECT_TRUE(meta_b);
  EXPECT_TRUE(meta_a->grad_.buffer());
  EXPECT_TRUE(meta_b->grad_.buffer());

  float *grad_a = meta_a->grad_.data<float>();
  float *grad_b = meta_b->grad_.data<float>();

  // Since c and d both depend on a and b, the gradients should accumulate to 2
  for (size_t i = 0; i < quasai::core::total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 2.0f);
    EXPECT_FLOAT_EQ(grad_b[i], 2.0f);
  }
}

TEST(AutoGradEngineTest, SubtractionBackward) {
  // Create tensors that require grad
  quasai::core::Tensor a =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  a.requires_grad(true);
  quasai::core::Tensor b =
      quasai::core::Tensor::zeros(quasai::core::Shape{2, 2});
  b.requires_grad(true);

  // Create a computation graph: c = a - b
  quasai::core::Tensor c = quasai::ops::sub(a, b);

  // Call backward on c
  EXPECT_NO_THROW(c.backward());

  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::autograd::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a);
  EXPECT_TRUE(meta_b);
  EXPECT_TRUE(meta_a->grad_.buffer());
  EXPECT_TRUE(meta_b->grad_.buffer());

  float *grad_a = meta_a->grad_.data<float>();
  float *grad_b = meta_b->grad_.data<float>();

  // Since c = a - b, the gradient for a should be ones and for b should be
  // -ones
  for (size_t i = 0; i < quasai::core::total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 1.0f);
    EXPECT_FLOAT_EQ(grad_b[i], -1.0f);
  }
}

TEST(AutoGradEngineTest, SharedIntermediateVisitedOnce) {
  // y = (x^2)^2 = x^4, so dy/dx = 4x^3. The intermediate x^2 is used by both
  // inputs of the outer multiplication and must only be backpropagated once.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor x = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 2}, quasai::core::DType::FLOAT32);
  x.requires_grad(true);

  quasai::core::Tensor x_squared = quasai::ops::mul(x, x);
  quasai::core::Tensor y = quasai::ops::mul(x_squared, x_squared);
  y.backward();

  const float *grad = x.autograd_meta()->grad_.data<float>();
  for (size_t i = 0; i < data.size(); ++i) {
    float expected = 4.0f * data[i] * data[i] * data[i];
    EXPECT_FLOAT_EQ(grad[i], expected);
  }
}

TEST(AutoGradEngineTest, BroadcastReductionGradient) {
  // L = sum(X + b) with b broadcast over the batch. dL/db must sum over rows.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor x = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 2}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor b = quasai::core::Tensor::zeros(
      quasai::core::Shape{1, 2}, quasai::core::DType::FLOAT32);
  b.requires_grad(true);

  quasai::core::Tensor loss = quasai::ops::sum(quasai::ops::add(x, b));
  loss.backward();

  const float *grad_b = b.autograd_meta()->grad_.data<float>();
  EXPECT_FLOAT_EQ(grad_b[0], 2.0f);
  EXPECT_FLOAT_EQ(grad_b[1], 2.0f);
}

TEST(AutoGradEngineTest, MeanDimBroadcastGradient) {
  // L = sum(mean(X + b, dim=1)) with 2 rows: each element contributes 1/2, so
  // dL/db = [1, 1].
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  quasai::core::Tensor x = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{2, 2}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor b = quasai::core::Tensor::zeros(
      quasai::core::Shape{1, 2}, quasai::core::DType::FLOAT32);
  b.requires_grad(true);

  quasai::core::Tensor loss =
      quasai::ops::sum(quasai::ops::mean(quasai::ops::add(x, b), 1));
  loss.backward();

  const float *grad_b = b.autograd_meta()->grad_.data<float>();
  EXPECT_FLOAT_EQ(grad_b[0], 1.0f);
  EXPECT_FLOAT_EQ(grad_b[1], 1.0f);
}
