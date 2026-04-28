#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/reduction_func.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::ops {

core::Tensor sum(const core::Tensor &a) {
  core::Tensor result =
      core::Tensor::empty(core::Shape{}, a.dtype(), a.device());

  const std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    autograd::Function *grad_fn = new autograd::SumFunction();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }

  switch (a.dtype()) {
    case core::DType::FLOAT32:
      do_sum<float>(a, result);
      break;
    case core::DType::FLOAT64:
      do_sum<double>(a, result);
      break;
    case core::DType::INT32:
      do_sum<int32_t>(a, result);
      break;
    case core::DType::INT64:
      do_sum<int64_t>(a, result);
      break;
    default:
      throw std::runtime_error("Unsupported data type for sum operation");
  }

  return result;
}

core::Tensor sum_to_shape(const core::Tensor &a, const core::Shape &target) {
  core::Shape a_shape = a.shape();
  size_t ndim_a = a_shape.dimensions();
  size_t ndim_t = target.dimensions();

  if (ndim_t > ndim_a) {
    throw std::runtime_error("Target shape cannot have more dimensions than "
                             "input tensor, got input" +
                             a_shape.to_string() + " and target " +
                             target.to_string());
  }

  core::Tensor out = core::Tensor::zeros(target, a.dtype(), a.device());

  const std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    autograd::Function *grad_fn = new autograd::SumToShapeFunction();
    grad_fn->inputs = {a};
    out.requires_grad(true);
    out.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }

  switch (a.dtype()) {
    case core::DType::FLOAT32:
      do_sum_to_shape<float>(a, out);
      break;
    case core::DType::FLOAT64:
      do_sum_to_shape<double>(a, out);
      break;
    case core::DType::INT32:
      do_sum_to_shape<int32_t>(a, out);
      break;
    case core::DType::INT64:
      do_sum_to_shape<int64_t>(a, out);
      break;
    default:
      throw std::runtime_error(
          "Unsupported data type for sum_to_shape operation");
  }

  return out;
}

core::Tensor broadcast_to_shape(const core::Tensor &a,
                                const core::Shape &target) {
  core::Shape a_shape = a.shape();
  size_t ndim_a = a_shape.dimensions();
  size_t ndim_t = target.dimensions();

  if (ndim_a > ndim_t) {
    throw std::runtime_error("Input shape cannot have more dimensions than "
                             "target shape, got input" +
                             a_shape.to_string() + " and target " +
                             target.to_string());
  }

  core::Tensor out = core::Tensor::empty(target, a.dtype(), a.device());

  const std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    autograd::Function *grad_fn = new autograd::BroadcastToShapeFunction();
    grad_fn->inputs = {a};
    out.requires_grad(true);
    out.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }

  switch (a.dtype()) {
    case core::DType::FLOAT32:
      do_broadcast_to_shape<float>(a, out);
      break;
    case core::DType::FLOAT64:
      do_broadcast_to_shape<double>(a, out);
      break;
    case core::DType::INT32:
      do_broadcast_to_shape<int32_t>(a, out);
      break;
    case core::DType::INT64:
      do_broadcast_to_shape<int64_t>(a, out);
      break;
    default:
      throw std::runtime_error(
          "Unsupported data type for broadcast_to_shape operation");
  }

  return out;
}

core::Tensor mean(const core::Tensor &a) {
  if (!core::is_floating(a.dtype())) {
    throw std::runtime_error(
        "Mean operation requires floating point data type");
  }

  core::Tensor result =
      core::Tensor::empty(core::Shape{}, a.dtype(), a.device());

  const std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();
  if (meta_a && meta_a->requires_grad) {
    autograd::Function *grad_fn = new autograd::MeanFunction();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<autograd::Function>(grad_fn));
  }

  switch (a.dtype()) {
    case core::DType::FLOAT32:
      do_sum<float>(a, result);
      do_unary_op<float>(result, result,
                         [num_elements = core::total_size(a.shape())](float x) {
                           return x / num_elements;
                         });
      break;
    case core::DType::FLOAT64:
      do_sum<double>(a, result);
      do_unary_op<double>(result, result,
                          [num_elements = core::total_size(a.shape())](
                              double x) { return x / num_elements; });
      break;
    default:
      throw std::runtime_error("Unsupported data type for mean operation");
  }

  return result;
}

} // namespace quasai::ops
