#include "quasai/autograd/engine.hpp"
#include "quasai/autograd/function.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

void AutoGradEngine::backward(const Tensor &tensor) {
  // If no grad provided, assume it's a scalar and create a grad of 1
  tensor.autograd_meta()->grad =
      Tensor::ones(Shape{}, tensor.dtype(), tensor.device());

  // Stack for DFS traversal of the computation graph
  std::vector<Tensor> stack{tensor};

  while (!stack.empty()) {
    Tensor current = stack.back();
    stack.pop_back();

    std::shared_ptr<AutoGradMeta> meta = current.autograd_meta();
    if (!meta || !meta->grad_fn) {
      LOG_DEBUG(
          "Reached leaf node in autograd graph, stopping backward traversal");
      continue; // No gradient function means we reached a leaf node
    }

    // Compute gradients for inputs
    Tensor current_grad = meta->grad;
    std::vector<Tensor> input_grads = meta->grad_fn->backward(current_grad);

    // Accumulate gradients for each input
    for (size_t i = 0; i < meta->grad_fn->inputs.size(); ++i) {
      const Tensor &input = meta->grad_fn->inputs[i];
      Tensor &input_grad = input_grads[i];

      std::shared_ptr<AutoGradMeta> input_meta = input.autograd_meta();
      if (input_meta) {
        if (input_meta->requires_grad) {
          if (input_meta->grad.buffer()) {
            // Accumulate gradient if it already exists
            input_meta->grad = add(input_meta->grad, input_grad);
          } else {
            // Otherwise, set the gradient for the first time
            input_meta->grad = input_grad;
          }
        }
      }

      // Add the input to the stack to compute its gradients
      stack.push_back(input);
    }
  }
}

} // namespace quasai
