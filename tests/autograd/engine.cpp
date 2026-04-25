#include "quasai/autograd/engine.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <gtest/gtest.h>

TEST(AutoGradEngineTest, BackwardNoGrad) {
  // Create a tensor that does not require grad
  quasai::Tensor tensor = quasai::Tensor::zeros(quasai::Shape{2, 2});

  // Call backward and ensure it does not throw
  EXPECT_NO_THROW(tensor.backward());
}

TEST(AutoGradEngineTest, BackwardSimpleGraph) {
  // Create tensors that require grad
  quasai::Tensor a = quasai::Tensor::zeros(quasai::Shape{2, 2});
  a.requires_grad(true);
  quasai::Tensor b = quasai::Tensor::zeros(quasai::Shape{2, 2});
  b.requires_grad(true);

  // Create a simple computation graph: c = a + b
  quasai::Tensor c = quasai::add(a, b);

  // Call backward on c
  EXPECT_NO_THROW(c.backward());

  // Check that gradients for a and b are created by engine the first time
  std::shared_ptr<quasai::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a->grad.buffer());
  EXPECT_TRUE(meta_b->grad.buffer());

  float *grad_a = meta_a->grad.data<float>();
  float *grad_b = meta_b->grad.data<float>();

  // Since c = a + b, the gradient for both a and b should be ones
  for (size_t i = 0; i < total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 1.0f);
    EXPECT_FLOAT_EQ(grad_b[i], 1.0f);
  }
}

TEST(AutoGradEngineTest, BackwardAccumulateGrad) {
  // Create tensors that require grad
  quasai::Tensor a = quasai::Tensor::zeros(quasai::Shape{2, 2});
  a.requires_grad(true);
  quasai::Tensor b = quasai::Tensor::zeros(quasai::Shape{2, 2});
  b.requires_grad(true);

  // Create a computation graph: c = a + b, d = a + b
  quasai::Tensor c = quasai::add(a, b);
  quasai::Tensor d = quasai::add(a, b);

  // Call backward on c and d to test gradient accumulation
  EXPECT_NO_THROW(c.backward());
  EXPECT_NO_THROW(d.backward());

  std::shared_ptr<quasai::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a);
  EXPECT_TRUE(meta_b);
  EXPECT_TRUE(meta_a->grad.buffer());
  EXPECT_TRUE(meta_b->grad.buffer());

  float *grad_a = meta_a->grad.data<float>();
  float *grad_b = meta_b->grad.data<float>();

  // Since c and d both depend on a and b, the gradients should accumulate to 2
  for (size_t i = 0; i < total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 2.0f);
    EXPECT_FLOAT_EQ(grad_b[i], 2.0f);
  }
}

TEST(AutoGradEngineTest, SubtractionBackward) {
  // Create tensors that require grad
  quasai::Tensor a = quasai::Tensor::zeros(quasai::Shape{2, 2});
  a.requires_grad(true);
  quasai::Tensor b = quasai::Tensor::zeros(quasai::Shape{2, 2});
  b.requires_grad(true);

  // Create a computation graph: c = a - b
  quasai::Tensor c = quasai::sub(a, b);

  // Call backward on c
  EXPECT_NO_THROW(c.backward());

  std::shared_ptr<quasai::AutoGradMeta> meta_a = a.autograd_meta();
  std::shared_ptr<quasai::AutoGradMeta> meta_b = b.autograd_meta();

  EXPECT_TRUE(meta_a);
  EXPECT_TRUE(meta_b);
  EXPECT_TRUE(meta_a->grad.buffer());
  EXPECT_TRUE(meta_b->grad.buffer());

  float *grad_a = meta_a->grad.data<float>();
  float *grad_b = meta_b->grad.data<float>();

  // Since c = a - b, the gradient for a should be ones and for b should be
  // -ones
  for (size_t i = 0; i < total_size(a.shape()); ++i) {
    EXPECT_FLOAT_EQ(grad_a[i], 1.0f);
    EXPECT_FLOAT_EQ(grad_b[i], -1.0f);
  }
}
