#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

// No new buffer allocation
// just create a new view with swapped shape and strides
Tensor transpose(const Tensor &a) {
  if (a.shape().dimensions() != 2) {
    throw std::runtime_error("transpose requires a 2D tensor");
  }

  TensorImpl impl_a_copy = a.get_impl_copy();

  impl_a_copy.shape = Shape{a.shape()[1], a.shape()[0]};
  impl_a_copy.strides = Strides{a.strides()[1], a.strides()[0]};
  impl_a_copy.is_contiguous = false;

  Tensor result = Tensor::from_impl(impl_a_copy);

  std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    Function *grad_fn = new TransposeFunction();
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  return result;
}

Tensor expand(const Tensor &a, const Shape &target) {
  if (target.dimensions() < a.shape().dimensions()) {
    throw std::runtime_error("expand requires target shape to have greater "
                             "than or equal number of "
                             "dimensions than input tensor");
  }

  TensorImpl impl_a_copy = a.get_impl_copy();

  Strides new_strides(target.dimensions());
  const Shape &a_shape = a.shape();
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

  return Tensor::from_impl(impl_a_copy);
}

Tensor reshape(const Tensor &a, const Shape &target) {
  if (total_size(a.shape()) != total_size(target)) {
    throw std::runtime_error("Total size of new shape must be the same as "
                             "the original shape, got " +
                             a.shape().to_string() + " and target " +
                             target.to_string());
  }
  // TODO: allocate new tensor which is contiguous and then reshape
  if (!a.is_contiguous()) {
    throw std::runtime_error("Reshape requires input tensor to be contiguous");
  }

  TensorImpl impl_a_copy = a.get_impl_copy();
  impl_a_copy.shape = target;
  impl_a_copy.strides = get_strides(target);

  Tensor result = Tensor::from_impl(impl_a_copy);

  std::shared_ptr<AutoGradMeta> meta_a = a.autograd_meta();

  if (meta_a && meta_a->requires_grad) {
    Function *grad_fn = new ReshapeFunction();
    grad_fn->inputs = {a};

    result.requires_grad(true);
    result.set_grad_fn(std::unique_ptr<Function>(grad_fn));
  }

  return result;
}

// Naive slice only by first dimension to keep contiguous memory layout and
// simple implementation
Tensor slice(const Tensor &a, size_t start, size_t end) {
  TensorImpl impl_a_copy = a.get_impl_copy();

  if (!a.is_contiguous()) {
    throw std::runtime_error("Slice requires input tensor to be contiguous");
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

  return Tensor::from_impl(impl_a_copy);
}

} // namespace quasai
