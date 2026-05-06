#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
#include <functional>

namespace quasai::autograd {
class Function;
} // namespace quasai::autograd

namespace quasai::ops {

/// @brief Helper to attach binary gradient function to a result tensor.
void add_binary_gradient(
    const core::Tensor &a, const core::Tensor &b, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);

/// @brief Element-wise addition of two tensors (with broadcasting).
core::Tensor add(const core::Tensor &a, const core::Tensor &b);
/// @brief Element-wise subtraction of two tensors (with broadcasting).
core::Tensor sub(const core::Tensor &a, const core::Tensor &b);
/// @brief Element-wise multiplication of two tensors (with broadcasting).
core::Tensor mul(const core::Tensor &a, const core::Tensor &b);
/// @brief Element-wise division of two tensors (with broadcasting).
core::Tensor div(const core::Tensor &a, const core::Tensor &b);

/// @brief Matrix multiplication of two 2D tensors.
core::Tensor matmul(const core::Tensor &a, const core::Tensor &b);

/// @brief Helper to attach unary gradient function to a result tensor.
void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);
/// @brief Element-wise negation.
core::Tensor neg(const core::Tensor &a);
/// @brief Element-wise absolute value.
core::Tensor abs(const core::Tensor &a);
/// @brief ReLU activation (max(0, x)).
core::Tensor relu(const core::Tensor &a);
/// @brief Heaviside step function (0 if x<0, 1 otherwise).
core::Tensor heaviside(const core::Tensor &a);
/// @brief Sign function (-1, 0, or 1).
core::Tensor signum(const core::Tensor &a);
/// @brief Sigmoid activation (1 / (1 + exp(-x))).
core::Tensor sigmoid(const core::Tensor &a);
/// @brief Hyperbolic tangent activation.
core::Tensor tanh(const core::Tensor &a);

/// @brief Sum all elements of a tensor.
core::Tensor sum(const core::Tensor &a);
/// @brief Sum elements and broadcast result to target shape.
core::Tensor sum_to_shape(const core::Tensor &a, const core::Shape &target);
/// @brief Mean of all elements.
core::Tensor mean(const core::Tensor &a);
/// @brief Mean along a dimension.
core::Tensor mean(const core::Tensor &a, int dim, bool keepdim = false);
/// @brief Element-wise power: a^exponent.
core::Tensor pow(const core::Tensor &a, float exponent);

/// @brief Transpose a 2D tensor (swap rows and columns).
core::Tensor transpose(const core::Tensor &a);
/// @brief View-based broadcast to target shape (no copy).
core::Tensor expand(const core::Tensor &a, const core::Shape &target);
/// @brief Reshape tensor to target shape (may copy if not contiguous).
core::Tensor reshape(const core::Tensor &a, const core::Shape &target);
/// @brief Make tensor contiguous in memory (copy if needed).
core::Tensor make_contiguous(const core::Tensor &a);
/// @brief Slice a 2D tensor along rows [start, end).
core::Tensor slice(const core::Tensor &a, size_t start, size_t end);

} // namespace quasai::ops
