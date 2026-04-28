#include "quasai/autograd/reduction_func.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai::autograd {

std::vector<core::Tensor>
SumFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("SumFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  core::Tensor grad_input = ops::broadcast_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<core::Tensor>
SumToShapeFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("SumToShapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  core::Tensor grad_input = ops::broadcast_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<core::Tensor>
BroadcastToShapeFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(("BroadcastToShapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  core::Tensor grad_input = ops::sum_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<core::Tensor>
MeanFunction::backward(const core::Tensor &grad_output) {
  const core::Tensor &input = inputs[0];
  LOG_DEBUG(
      ("MeanFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  switch (input.dtype()) {
    case core::DType::FLOAT32: {
      const float num_elements =
          static_cast<float>(core::total_size(input.shape()));
      core::Tensor num_elements_tensor = core::Tensor::from_data(
          &num_elements, core::Shape{}, input.dtype(), input.device());
      core::Tensor grad_input = ops::broadcast_to_shape(
          ops::div(grad_output, num_elements_tensor), input.shape());
      LOG_DEBUG(
          ("Final shape of grad_input = " + grad_input.shape().to_string())
              .c_str());
      return {grad_input};
    }
    case core::DType::FLOAT64: {
      const double num_elements =
          static_cast<double>(core::total_size(input.shape()));
      core::Tensor num_elements_tensor = core::Tensor::from_data(
          &num_elements, core::Shape{}, input.dtype(), input.device());
      core::Tensor grad_input = ops::broadcast_to_shape(
          ops::div(grad_output, num_elements_tensor), input.shape());
      return {grad_input};
    }
    default:
      LOG_DEBUG("MeanFunction backward: input dtype = UNKNOWN");
  }
  return {core::Tensor()};
}

} // namespace quasai::autograd
