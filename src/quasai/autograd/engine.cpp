#include "quasai/autograd/engine.hpp"

#include "quasai/autograd/function.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

// Performs backward pass starting from `tensor`.
// Initializes the output gradient to ones, then traverses the compute graph
// using a stack-based DFS. For each node with a grad_fn, calls backward()
// to get input gradients, accumulates them on input tensors that require grad,
// and continues traversal until all leaf nodes are reached.
void AutoGradEngine::backward(const core::Tensor &tensor) {
  if (!tensor_requires_grad(tensor)) {
    LOG_DEBUG("No gradients to compute since requires_grad is false");
    return;
  }

  tensor.autograd_meta()->grad_ =
      core::Tensor::ones(tensor.shape(), tensor.dtype(), tensor.device());

  std::vector<core::Tensor> stack{tensor};

  while (!stack.empty()) {
    core::Tensor current = stack.back();
    stack.pop_back();

    if (!tensor_requires_grad(current)) {
      LOG_DEBUG(
          "Reached leaf node in autograd graph, stopping backward traversal");
      continue;
    }

    std::shared_ptr<AutoGradMeta> meta = current.autograd_meta();
    if (!meta || !meta->grad_fn_) {
      continue;
    }

    core::Tensor current_grad = meta->grad_;
    std::vector<core::Tensor> input_grads =
        meta->grad_fn_->backward(current_grad);

    for (size_t i = 0; i < meta->grad_fn_->inputs_.size(); ++i) {
      const core::Tensor &input = meta->grad_fn_->inputs_[i];
      core::Tensor &input_grad = input_grads[i];

      std::shared_ptr<AutoGradMeta> input_meta = input.autograd_meta();
      if (input_meta) {
        if (input_meta->requires_grad_) {
          if (input_meta->grad_.buffer()) {
            input_meta->grad_ = ops::add(input_meta->grad_, input_grad);
          } else {
            input_meta->grad_ = input_grad;
          }
        }
      }

      stack.push_back(input);
    }
  }
}

} // namespace quasai::autograd
