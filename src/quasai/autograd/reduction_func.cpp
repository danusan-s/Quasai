#include "quasai/autograd/function.hpp"
#include "quasai/ops/tensor_ops.hpp"

namespace quasai {

std::vector<Tensor> SumFunction::backward(const Tensor &grad_output) {
  // grad for input is just grad_output broadcasted to input shape
  const Tensor &input = inputs[0];
  LOG_DEBUG(("SumFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  Tensor grad_input = broadcast_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<Tensor> SumToShapeFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output broadcasted to input shape
  const Tensor &input = inputs[0];
  LOG_DEBUG(("SumToShapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  Tensor grad_input = broadcast_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<Tensor>
BroadcastToShapeFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output summed to input shape
  const Tensor &input = inputs[0];
  LOG_DEBUG(("BroadcastToShapeFunction backward: grad_output shape = " +
             grad_output.shape().to_string() +
             ", input shape = " + input.shape().to_string())
                .c_str());
  Tensor grad_input = sum_to_shape(grad_output, input.shape());
  return {grad_input};
}

std::vector<Tensor> MeanFunction::backward(const Tensor &grad_output) {
  // grad for input is grad_output broadcasted to input shape and divided by
  // number of elements in input
  const Tensor &input = inputs[0];
  LOG_DEBUG(
      ("MeanFunction backward: input shape = " + input.shape().to_string() +
       ", grad_output shape = " + grad_output.shape().to_string())
          .c_str());
  switch (input.dtype()) {
    case DType::FLOAT32: {
      const float num_elements = static_cast<float>(total_size(input.shape()));
      Tensor num_elements_tensor = Tensor::from_data(
          &num_elements, Shape{}, input.dtype(), input.device());
      Tensor grad_input = broadcast_to_shape(
          div(grad_output, num_elements_tensor), input.shape());
      LOG_DEBUG(
          ("Final shape of grad_input = " + grad_input.shape().to_string())
              .c_str());
      return {grad_input};
      break;
    }
    case DType::FLOAT64: {
      const double num_elements =
          static_cast<double>(total_size(input.shape()));
      Tensor num_elements_tensor = Tensor::from_data(
          &num_elements, Shape{}, input.dtype(), input.device());
      Tensor grad_input = broadcast_to_shape(
          div(grad_output, num_elements_tensor), input.shape());
      return {grad_input};
      break;
    }
    default:
      LOG_DEBUG("MeanFunction backward: input dtype = UNKNOWN");
  }
  return {Tensor()};
}

} // namespace quasai
