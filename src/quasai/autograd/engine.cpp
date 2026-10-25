#include "quasai/autograd/engine.hpp"

#include "quasai/autograd/function.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/ops/tensor_ops.hpp"

#include <algorithm>
#include <vector>

namespace quasai::autograd {

namespace {

struct VisitFrame {
  core::Tensor tensor;
  bool expanded;
};

// Builds a post-order of the reachable autograd graph: every tensor appears
// after all of its inputs. Reversing this order yields an order in which each
// tensor is visited only after every consumer of its gradient has been
// processed. Visited nodes are tracked by AutoGradMeta identity so that shared
// subgraphs (e.g. residual connections) are emitted only once.
std::vector<core::Tensor> build_postorder(const core::Tensor &root) {
  std::vector<core::Tensor> postorder;
  std::vector<AutoGradMeta *> visited;
  visited.reserve(32); // Typical graph size
  postorder.reserve(32);

  std::vector<VisitFrame> stack;
  stack.reserve(32);
  stack.push_back({root, false});

  while (!stack.empty()) {
    VisitFrame &frame = stack.back();
    std::shared_ptr<AutoGradMeta> meta = frame.tensor.autograd_meta();

    if (!meta || !tensor_requires_grad(frame.tensor)) {
      stack.pop_back();
      continue;
    }

    AutoGradMeta *id = meta.get();
    if (frame.expanded) {
      postorder.push_back(frame.tensor);
      stack.pop_back();
      continue;
    }

    // Linear search for visited - faster than unordered_set for small graphs
    bool found = false;
    for (AutoGradMeta *v : visited) {
      if (v == id) {
        found = true;
        break;
      }
    }
    if (found) {
      stack.pop_back();
      continue;
    }
    visited.push_back(id);

    frame.expanded = true;
    if (meta->grad_fn_) {
      for (const core::Tensor &input : meta->grad_fn_->inputs_) {
        stack.push_back({input, false});
      }
    }
  }

  return postorder;
}

} // namespace

// Performs a backward pass starting from `tensor`.
// Initializes the output gradient to ones, builds a topological ordering of the
// compute graph, then walks it in reverse so that each node's gradient is fully
// accumulated before it is propagated to its inputs.
void AutoGradEngine::backward(const core::Tensor &tensor) {
  if (!tensor_requires_grad(tensor)) {
    LOG_DEBUG("No gradients to compute since requires_grad is false");
    return;
  }

  tensor.autograd_meta()->grad_ =
      core::Tensor::ones(tensor.shape(), tensor.dtype(), tensor.device());

  std::vector<core::Tensor> postorder = build_postorder(tensor);

  // Reverse post-order: consumers before producers.
  for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
    const core::Tensor &current = *it;
    std::shared_ptr<AutoGradMeta> meta = current.autograd_meta();
    if (!meta || !meta->grad_fn_ || !meta->grad_.buffer()) {
      continue;
    }

    std::vector<core::Tensor> input_grads =
        meta->grad_fn_->backward(meta->grad_);

    for (size_t i = 0; i < meta->grad_fn_->inputs_.size(); ++i) {
      const core::Tensor &input = meta->grad_fn_->inputs_[i];
      core::Tensor &input_grad = input_grads[i];

      std::shared_ptr<AutoGradMeta> input_meta = input.autograd_meta();
      if (!input_meta || !input_meta->requires_grad_ || !input_grad.buffer()) {
        continue;
      }

      if (input_meta->grad_.buffer()) {
        input_meta->grad_ = ops::add(input_meta->grad_, input_grad);
      } else {
        input_meta->grad_ = std::move(input_grad);
      }
    }
  }
}

} // namespace quasai::autograd
