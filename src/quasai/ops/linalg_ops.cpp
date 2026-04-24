#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.shape().dimensions() != 2 || b.shape().dimensions() != 2) {
    throw std::runtime_error("matmul requires 2D tensors");
  }
  if (a.shape()[1] != b.shape()[0]) {
    throw std::runtime_error("Inner dimensions must match for matmul");
  }

  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  Shape result_shape{M, N};
  Tensor result = Tensor::empty(result_shape, a.dtype(), a.device());

  const std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  const std::shared_ptr<AutoGradMeta> meta_b = b.autograd_meta();

  if (meta_a && meta_a->requires_grad && meta_b && meta_b->requires_grad) {
    MatMulFunction *grad_fn = new MatMulFunction();
    grad_fn->inputs = {a, b};

    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(
        "Data types of input tensors must match for matmul operation");
  }

  switch (a.dtype()) {
    case DType::FLOAT32:
      do_matmul<float>(a, b, result);
      break;
    case DType::FLOAT64:
      do_matmul<double>(a, b, result);
      break;
    case DType::INT32:
      do_matmul<int32_t>(a, b, result);
      break;
    case DType::INT64:
      do_matmul<int64_t>(a, b, result);
      break;
    default:
      throw std::runtime_error("Unsupported data type for matmul operation");
  }

  return result;
}

// No new buffer allocation
// just create a new view with swapped shape and strides
Tensor transpose(const Tensor &a) {
  if (a.shape().dimensions() != 2) {
    throw std::runtime_error("transpose requires a 2D tensor");
  }

  TensorImpl impl_a_copy = a.get_impl_copy();

  impl_a_copy.shape = Shape{a.shape()[1], a.shape()[0]};
  impl_a_copy.strides = Strides{a.strides()[1], a.strides()[0]};

  return Tensor::from_impl(impl_a_copy);
}

} // namespace quasai
