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
    auto grad_fn = std::make_unique<autograd::SumFunction>();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::move(grad_fn));
  }

  dispatch_by_dtype(a.dtype(), [&]<typename T>() { do_sum<T>(a, result); });

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
    auto grad_fn = std::make_unique<autograd::SumToShapeFunction>();
    grad_fn->inputs = {a};
    out.requires_grad(true);
    out.set_grad_fn(std::move(grad_fn));
  }

  dispatch_by_dtype(a.dtype(),
                    [&]<typename T>() { do_sum_to_shape<T>(a, out); });

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
    auto grad_fn = std::make_unique<autograd::MeanFunction>();
    grad_fn->inputs = {a};
    result.requires_grad(true);
    result.set_grad_fn(std::move(grad_fn));
  }

  dispatch_by_dtype(a.dtype(), [&]<typename T>() {
    do_sum<T>(a, result);
    do_unary_op<T>(result, result,
                   [num_elements = core::total_size(a.shape())](T x) {
                     return x / static_cast<T>(num_elements);
                   });
  });

  return result;
}

} // namespace quasai::ops
