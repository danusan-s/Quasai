#pragma once

#include "quasai/core/dtype.hpp"
#include "quasai/core/tensor.hpp"
#include "quasai/nn/parameter.hpp"
#include "quasai/utils/random.hpp"

namespace quasai::nn {

enum class Initialization {
  ZEROS,
  ONES,
  HE_UNIFORM,
  HE_NORMAL,
  GLOROT_UNIFORM,
  GLOROT_NORMAL
};

inline std::string to_string(Initialization init) {
  switch (init) {
    case Initialization::ZEROS:
      return "zeros";
    case Initialization::ONES:
      return "ones";
    case Initialization::HE_UNIFORM:
      return "he_uniform";
    case Initialization::HE_NORMAL:
      return "he_normal";
    case Initialization::GLOROT_UNIFORM:
      return "glorot_uniform";
    case Initialization::GLOROT_NORMAL:
      return "glorot_normal";
    default:
      return "unknown";
  }
}

Parameter zeros(const core::Shape &shape,
                core::DType dtype = core::DType::FLOAT32,
                core::Device device = core::Device::cpu());

Parameter ones(const core::Shape &shape,
               core::DType dtype = core::DType::FLOAT32,
               core::Device device = core::Device::cpu());

template <typename T> void uniform(Parameter &t, T low, T high) {
  std::uniform_real_distribution<T> dist(low, high);
  t.check_valid_dtype<T>();
  T *data = static_cast<T *>(t.buffer()->raw_data());
  size_t num_elements = total_size(t.shape());
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = dist(RNG::instance().engine());
  }
}

template <typename T> void normal(Parameter &t, T mean, T stddev) {
  std::normal_distribution<T> dist(mean, stddev);
  t.check_valid_dtype<T>();
  T *data = static_cast<T *>(t.buffer()->raw_data());
  size_t num_elements = total_size(t.shape());
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = dist(RNG::instance().engine());
  }
}

void uniform(Parameter &t, float low, float high);

void normal(Parameter &t, float mean, float stddev);

Parameter he_uniform(const core::Shape &shape,
                     core::DType dtype = core::DType::FLOAT32,
                     core::Device device = core::Device::cpu());

Parameter he_normal(const core::Shape &shape,
                    core::DType dtype = core::DType::FLOAT32,
                    core::Device device = core::Device::cpu());

Parameter glorot_uniform(const core::Shape &shape,
                         core::DType dtype = core::DType::FLOAT32,
                         core::Device device = core::Device::cpu());

Parameter glorot_normal(const core::Shape &shape,
                        core::DType dtype = core::DType::FLOAT32,
                        core::Device device = core::Device::cpu());

Parameter initialize(const core::Shape &shape, core::DType dtype,
                     core::Device device, Initialization init);

} // namespace quasai::nn
