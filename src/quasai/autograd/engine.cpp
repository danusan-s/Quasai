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
  if (!tensor.autograd_meta() || !tensor.autograd_meta()->requires_grad) {
    LOG_DEBUG("No gradients to compute since requires_grad is false");
    return;
  }

  tensor.autograd_meta()->grad =
      core::Tensor::ones(tensor.shape(), tensor.dtype(), tensor.device());

  std::vector<core::Tensor> stack{tensor};

  while (!stack.empty()) {
    core::Tensor current = stack.back();
    stack.pop_back();

    std::shared_ptr<AutoGradMeta> meta = current.autograd_meta();
    if (!meta || !meta->grad_fn) {
      LOG_DEBUG(
          "Reached leaf node in autograd graph, stopping backward traversal");
      continue;
    }

    core::Tensor current_grad = meta->grad;
    std::vector<core::Tensor> input_grads =
        meta->grad_fn->backward(current_grad);

    for (size_t i = 0; i < meta->grad_fn->inputs.size(); ++i) {
      const core::Tensor &input = meta->grad_fn->inputs[i];
      core::Tensor &input_grad = input_grads[i];

      std::shared_ptr<AutoGradMeta> input_meta = input.autograd_meta();
      if (input_meta) {
        if (input_meta->requires_grad) {
          if (input_meta->grad.buffer()) {
            input_meta->grad = ops::add(input_meta->grad, input_grad);
          } else {
            input_meta->grad = input_grad;
          }
        }
      }

      stack.push_back(input);
    }
  }
}

} // namespace quasai::autograd
