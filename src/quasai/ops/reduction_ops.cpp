#include "quasai/autograd/metadata.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

Tensor sum(const Tensor &a) {
  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, a.dtype(), a.device());

  const std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    // Create a gradient function for sum that will compute the gradient
    Function *grad_fn = new SumFunction();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  switch (a.dtype()) {
    case DType::FLOAT32:
      do_sum<float>(a, result);
      break;
    case DType::FLOAT64:
      do_sum<double>(a, result);
      break;
    case DType::INT32:
      do_sum<int32_t>(a, result);
      break;
    case DType::INT64:
      do_sum<int64_t>(a, result);
      break;
    default:
      throw std::runtime_error("Unsupported data type for sum operation");
  }

  return result;
}

// Reduce tensor a to target shape by summing over the appropriate dimensions
Tensor sum_to_shape(const Tensor &a, const Shape &target) {
  Shape a_shape = a.shape();
  size_t ndim_a = a_shape.dimensions();
  size_t ndim_t = target.dimensions();

  if (ndim_t > ndim_a) {
    throw std::runtime_error("Target shape cannot have more dimensions than "
                             "input tensor, got input" +
                             a_shape.to_string() + " and target " +
                             target.to_string());
  }

  Tensor out = Tensor::zeros(target, a.dtype(), a.device());

  const std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    Function *grad_fn = new SumToShapeFunction();
    grad_fn->inputs = {a};
    out.requires_grad(true);
    out.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  switch (a.dtype()) {
    case DType::FLOAT32:
      do_sum_to_shape<float>(a, out);
      break;
    case DType::FLOAT64:
      do_sum_to_shape<double>(a, out);
      break;
    case DType::INT32:
      do_sum_to_shape<int32_t>(a, out);
      break;
    case DType::INT64:
      do_sum_to_shape<int64_t>(a, out);
      break;
    default:
      throw std::runtime_error(
          "Unsupported data type for sum_to_shape operation");
  }

  return out;
}

// Expand tensor a to target shape by broadcasting
Tensor broadcast_to_shape(const Tensor &a, const Shape &target) {
  Shape a_shape = a.shape();
  size_t ndim_a = a_shape.dimensions();
  size_t ndim_t = target.dimensions();

  if (ndim_a > ndim_t) {
    throw std::runtime_error("Input shape cannot have more dimensions than "
                             "target shape, got input" +
                             a_shape.to_string() + " and target " +
                             target.to_string());
  }

  Tensor out = Tensor::empty(target, a.dtype(), a.device());

  const std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    Function *grad_fn = new BroadcastToShapeFunction();
    grad_fn->inputs = {a};
    out.requires_grad(true);
    out.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  switch (a.dtype()) {
    case DType::FLOAT32:
      do_broadcast_to_shape<float>(a, out);
      break;
    case DType::FLOAT64:
      do_broadcast_to_shape<double>(a, out);
      break;
    case DType::INT32:
      do_broadcast_to_shape<int32_t>(a, out);
      break;
    case DType::INT64:
      do_broadcast_to_shape<int64_t>(a, out);
      break;
    default:
      throw std::runtime_error(
          "Unsupported data type for broadcast_to_shape operation");
  }

  return out;
}

Tensor mean(const Tensor &a) {
  if (!is_floating(a.dtype())) {
    throw std::runtime_error(
        "Mean operation requires floating point data type");
  }

  // Scalar result so empty shape
  Tensor result = Tensor::empty(Shape{}, a.dtype(), a.device());

  const std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    // Create a gradient function for mean that will compute the gradient of the
    Function *grad_fn = new MeanFunction();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  switch (a.dtype()) {
    case DType::FLOAT32:
      do_sum<float>(a, result);
      do_unary_op<float>(result, result,
                         [num_elements = total_size(a.shape())](float x) {
                           return x / num_elements;
                         });
      break;
    case DType::FLOAT64:
      do_sum<double>(a, result);
      do_unary_op<double>(result, result,
                          [num_elements = total_size(a.shape())](double x) {
                            return x / num_elements;
                          });
      break;
    default:
      throw std::runtime_error("Unsupported data type for mean operation");
  }

  return result;
}

}; // namespace quasai
