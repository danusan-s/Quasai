#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/reduction_func.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/ops/cpu_kernel.hpp"
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

  size_t num_elements = total_size(a.shape());

  core::Tensor sum_result = sum(a);
  core::Tensor num_elements_tensor =
      core::Tensor::from_scalar(num_elements, a.dtype(), a.device());
  result = div(sum_result, num_elements_tensor);

  return result;
}

core::Tensor mean(const core::Tensor &a, int64_t dim, bool keepdim) {
  if (!core::is_floating(a.dtype())) {
    throw std::runtime_error(
        "Mean operation requires floating point data type");
  }

  core::Shape out_shape = a.shape();
  if (dim < 0 || static_cast<size_t>(dim) >= out_shape.dimensions()) {
    throw std::runtime_error("Dimension out of range for mean operation");
  }
  size_t num_elements = out_shape[dim];
  out_shape[dim] = 1;

  core::Tensor result = core::Tensor::empty(out_shape, a.dtype(), a.device());

  core::Tensor sum_result = sum_to_shape(a, out_shape);

  core::Tensor num_elements_tensor =
      core::Tensor::from_scalar(num_elements, a.dtype(), a.device());
  core::Tensor mean_result = div(sum_result, num_elements_tensor);

  if (!keepdim) {
    result = reshape(mean_result, squeeze_shape(result.shape(), dim));
  }

  return result;
}

} // namespace quasai::ops
