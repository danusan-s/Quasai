#include "quasai/nn/init.hpp"
#include "quasai/utils/logger.hpp"

namespace quasai::nn {

Parameter zeros(const core::Shape &shape, core::DType dtype, core::Device device) {
  return Parameter(core::Tensor::zeros(shape, dtype, device));
}

Parameter ones(const core::Shape &shape, core::DType dtype, core::Device device) {
  return Parameter(core::Tensor::ones(shape, dtype, device));
}

void uniform(Parameter &t, float low, float high) {
  switch (t.dtype()) {
    case core::DType::FLOAT32:
      uniform<float>(t, low, high);
      break;
    case core::DType::FLOAT64:
      uniform<double>(t, low, high);
      break;
    default:
      throw std::runtime_error("Unsupported data type for uniform");
  }
}

void normal(Parameter &t, float mean, float stddev) {
  switch (t.dtype()) {
    case core::DType::FLOAT32:
      normal<float>(t, mean, stddev);
      break;
    case core::DType::FLOAT64:
      normal<double>(t, mean, stddev);
      break;
    default:
      throw std::runtime_error("Unsupported data type for normal");
  }
}

Parameter he_uniform(const core::Shape &shape, core::DType dtype, core::Device device) {
  if (shape.dimensions() < 2) {
    throw std::runtime_error("he_uniform requires at least 2D tensor");
  }

  int64_t fan_in = shape[shape.dimensions() - 2];
  float limit = std::sqrt(6.0f / fan_in);
  Parameter t = zeros(shape, dtype, device);
  uniform(t, -limit, limit);
  return t;
}

Parameter he_normal(const core::Shape &shape, core::DType dtype, core::Device device) {
  if (shape.dimensions() < 2) {
    throw std::runtime_error("he_normal requires at least 2D tensor");
  }

  int64_t fan_in = shape[shape.dimensions() - 2];
  float stddev = std::sqrt(2.0f / fan_in);
  Parameter t = zeros(shape, dtype, device);
  normal(t, 0.0f, stddev);
  return t;
}

Parameter glorot_uniform(const core::Shape &shape, core::DType dtype, core::Device device) {
  if (shape.dimensions() < 2) {
    throw std::runtime_error("glorot_uniform requires at least 2D tensor");
  }

  int64_t fan_in = shape[shape.dimensions() - 2];
  int64_t fan_out = shape[shape.dimensions() - 1];
  float limit = std::sqrt(6.0f / (fan_in + fan_out));
  Parameter t = zeros(shape, dtype, device);
  uniform(t, -limit, limit);
  return t;
}

Parameter glorot_normal(const core::Shape &shape, core::DType dtype, core::Device device) {
  if (shape.dimensions() < 2) {
    throw std::runtime_error("glorot_normal requires at least 2D tensor");
  }

  int64_t fan_in = shape[shape.dimensions() - 2];
  int64_t fan_out = shape[shape.dimensions() - 1];
  float stddev = std::sqrt(2.0f / (fan_in + fan_out));
  Parameter t = zeros(shape, dtype, device);
  normal(t, 0.0f, stddev);
  return t;
}

Parameter initialize(const core::Shape &shape, core::DType dtype, core::Device device,
                     Initialization init) {
  LOG_DEBUG(("Initializing tensor with init: " + to_string(init)).c_str());
  switch (init) {
    case Initialization::ZEROS:
      return zeros(shape, dtype, device);
    case Initialization::ONES:
      return ones(shape, dtype, device);
    case Initialization::HE_UNIFORM:
      return he_uniform(shape, dtype, device);
    case Initialization::HE_NORMAL:
      return he_normal(shape, dtype, device);
    case Initialization::GLOROT_UNIFORM:
      return glorot_uniform(shape, dtype, device);
    case Initialization::GLOROT_NORMAL:
      return glorot_normal(shape, dtype, device);
    default:
      throw std::runtime_error("Unknown initialization method");
  }
}

} // namespace quasai::nn
