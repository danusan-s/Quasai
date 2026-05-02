#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
#include <functional>

namespace quasai::autograd {
class Function;
} // namespace quasai::autograd

namespace quasai::ops {

// Binary ops
void add_binary_gradient(
    const core::Tensor &a, const core::Tensor &b, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);
core::Tensor add(const core::Tensor &a, const core::Tensor &b);
core::Tensor sub(const core::Tensor &a, const core::Tensor &b);
core::Tensor mul(const core::Tensor &a, const core::Tensor &b);
core::Tensor div(const core::Tensor &a, const core::Tensor &b);

core::Tensor matmul(const core::Tensor &a, const core::Tensor &b);

// Unary ops
void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);
core::Tensor neg(const core::Tensor &a);
core::Tensor abs(const core::Tensor &a);
core::Tensor relu(const core::Tensor &a);
core::Tensor heaviside(const core::Tensor &a);
core::Tensor signum(const core::Tensor &a);
core::Tensor sigmoid(const core::Tensor &a);
core::Tensor tanh(const core::Tensor &a);

// Reduction ops
core::Tensor sum(const core::Tensor &a);
core::Tensor sum_to_shape(const core::Tensor &a, const core::Shape &target);
core::Tensor mean(const core::Tensor &a);
core::Tensor mean(const core::Tensor &a, int dim, bool keepdim = false);
core::Tensor pow(const core::Tensor &a, float exponent);

// View ops
core::Tensor transpose(const core::Tensor &a);
core::Tensor expand(const core::Tensor &a,
                    const core::Shape &target); // broadcast_to_shape but view
                                                // and no new buffer allocation
core::Tensor reshape(const core::Tensor &a, const core::Shape &target);
core::Tensor make_contiguous(const core::Tensor &a);
core::Tensor slice(const core::Tensor &a, size_t start, size_t end);

} // namespace quasai::ops
