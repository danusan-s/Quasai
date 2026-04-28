#include "quasai/autograd/metadata.hpp"
#include "quasai/autograd/view_func.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include "quasai/utils/logger.hpp"

namespace quasai::ops {

inline void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor) {
  std::shared_ptr<autograd::AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    auto grad_fn = grad_fn_constructor();
    if (!grad_fn) {
      throw std::runtime_error("Gradient function constructor returned nullptr "
                               "or not implemented for this operation");
    }
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::move(grad_fn));
  }
}

core::Tensor transpose(const core::Tensor &a) {
  if (a.shape().dimensions() != 2) {
    throw std::runtime_error("transpose requires a 2D tensor");
  }

  core::TensorImpl impl_a_copy = a.get_impl_copy();

  impl_a_copy.shape = core::Shape{a.shape()[1], a.shape()[0]};
  impl_a_copy.strides = core::Strides{a.strides()[1], a.strides()[0]};
  impl_a_copy.is_contiguous = false;

  core::Tensor result = core::Tensor::from_impl(impl_a_copy);

  add_unary_gradient(a, result, []() {
    return std::make_unique<autograd::TransposeFunction>();
  });

  return result;
}

core::Tensor expand(const core::Tensor &a, const core::Shape &target) {
  if (target.dimensions() < a.shape().dimensions()) {
    throw std::runtime_error("expand requires target shape to have greater "
                             "than or equal number of "
                             "dimensions than input tensor");
  }

  core::TensorImpl impl_a_copy = a.get_impl_copy();

  core::Strides new_strides(target.dimensions());
  const core::Shape &a_shape = a.shape();
  const std::size_t a_dims = a_shape.dimensions();

  for (size_t i = 0; i < target.dimensions(); ++i) {
    if (i < a_dims && a_shape[a_dims - 1 - i] != 1) {
      if (a_shape[a_dims - 1 - i] == target[target.dimensions() - 1 - i]) {
        new_strides[target.dimensions() - 1 - i] =
            impl_a_copy.strides[a_dims - 1 - i];
      } else {
        throw std::runtime_error(
            "Cannot expand dimension " + std::to_string(a_dims - i) +
            " of input tensor from size " +
            std::to_string(a_shape[a_dims - 1 - i]) + " to size " +
            std::to_string(target[target.dimensions() - 1 - i]));
      }
    } else {
      new_strides[target.dimensions() - 1 - i] = 0;
    }
  }

  impl_a_copy.shape = target;
  impl_a_copy.strides = new_strides;
  impl_a_copy.is_contiguous = false;

  core::Tensor result = core::Tensor::from_impl(impl_a_copy);

  add_unary_gradient(
      a, result, []() { return std::make_unique<autograd::ExpandFunction>(); });
  return result;
}

core::Tensor make_contiguous(const core::Tensor &a) {
  if (a.is_contiguous()) {
    return a;
  }

  core::Tensor result = core::Tensor::empty(a.shape(), a.dtype(), a.device());

  size_t num_elements = core::total_size(a.shape());

  dispatch_by_dtype(a.dtype(),
                    [&]<typename T>() { do_contiguous_copy<T>(a, result); });

  add_unary_gradient(a, result, []() {
    return std::make_unique<autograd::MakeContiguousFunction>();
  });

  return result;
}

core::Tensor reshape(const core::Tensor &a, const core::Shape &target) {
  if (core::total_size(a.shape()) != core::total_size(target)) {
    throw std::runtime_error("Total size of new shape must be the same as "
                             "the original shape, got " +
                             a.shape().to_string() + " and target " +
                             target.to_string());
  }
  core::TensorImpl impl_a_copy = a.get_impl_copy();
  if (!a.is_contiguous()) {
    LOG_WARNING("Input tensor is not contiguous, making it contiguous before "
                "reshaping");
    impl_a_copy = make_contiguous(a).get_impl_copy();
  }
  impl_a_copy.shape = target;
  impl_a_copy.strides = core::get_strides(target);

  core::Tensor result = core::Tensor::from_impl(impl_a_copy);

  add_unary_gradient(a, result, []() {
    return std::make_unique<autograd::ReshapeFunction>();
  });

  return result;
}

core::Tensor slice(const core::Tensor &a, size_t start, size_t end) {
  core::TensorImpl impl_a_copy = a.get_impl_copy();

  if (!a.is_contiguous()) {
    LOG_WARNING(
        "Input tensor is not contiguous, making it contiguous before slicing");
    impl_a_copy = make_contiguous(a).get_impl_copy();
  }

  if (a.shape().dimensions() == 0) {
    throw std::runtime_error("Slice requires input tensor to have at least 1 "
                             "dimension");
  }

  if (start >= end || end > a.shape()[0]) {
    throw std::runtime_error("Invalid slice range: [" + std::to_string(start) +
                             ", " + std::to_string(end) + ")");
  }

  impl_a_copy.shape[0] = end - start;
  impl_a_copy.offset += start * impl_a_copy.strides[0];

  return core::Tensor::from_impl(impl_a_copy);
}

} // namespace quasai::ops
