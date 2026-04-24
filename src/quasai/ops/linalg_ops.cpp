#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.shape().dimensions() > 2 || b.shape().dimensions() > 2) {
    throw std::runtime_error("matmul requires 2D tensors");
  }
  if (a.shape().dimensions() == 0 || b.shape().dimensions() == 0) {
    throw std::runtime_error("matmul requires non-scalar tensors");
  }

  if (a.shape().dimensions() == 1) {
    // If a is 1D, treat it as a row vector
    LOG_DEBUG(("matmul: treating a as row vector with shape (1, " +
               std::to_string(a.shape()[0]) + ")")
                  .c_str());
    Tensor a_reshaped = reshape(a, Shape{1, a.shape()[0]});
    return matmul(a_reshaped, b);
  }
  if (b.shape().dimensions() == 1) {
    // If b is 1D, treat it as a column vector
    LOG_DEBUG(("matmul: treating b as column vector with shape (" +
               std::to_string(b.shape()[0]) + ", 1)")
                  .c_str());
    Tensor b_reshaped = reshape(b, Shape{b.shape()[0], 1});
    return matmul(a, b_reshaped);
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

} // namespace quasai
