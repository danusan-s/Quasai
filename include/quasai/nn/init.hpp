#pragma once

#include "quasai/core/dtype.hpp"
#include "quasai/core/tensor.hpp"
#include "quasai/nn/parameter.hpp"
#include "quasai/utils/random.hpp"

namespace quasai::nn {

/// @brief Initialization schemes for tensor parameters.
enum class Initialization {
  ZEROS,          ///< Initialize with zeros
  ONES,           ///< Initialize with ones
  HE_UNIFORM,     ///< He uniform initialization
  HE_NORMAL,      ///< He normal initialization
  GLOROT_UNIFORM, ///< Glorot (Xavier) uniform initialization
  GLOROT_NORMAL   ///< Glorot (Xavier) normal initialization
};

/**
 * @brief Convert Initialization enum to string.
 * @param init Initialization scheme.
 * @return String representation.
 */
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

/**
 * @brief Initialize a parameter with zeros.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with zeros.
 */
Parameter zeros(const core::Shape &shape,
                core::DType dtype = core::DType::FLOAT32,
                core::Device device = core::Device::cpu());

/**
 * @brief Initialize a parameter with ones.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with ones.
 */
Parameter ones(const core::Shape &shape,
               core::DType dtype = core::DType::FLOAT32,
               core::Device device = core::Device::cpu());

/**
 * @brief Fill a parameter with values drawn from a uniform distribution.
 * @tparam T Data type of the parameter.
 * @param t Parameter to fill.
 * @param low Lower bound of the distribution.
 * @param high Upper bound of the distribution.
 */
template <typename T> void uniform(Parameter &t, T low, T high) {
  std::uniform_real_distribution<T> dist(low, high);
  t.check_valid_dtype<T>();
  T *data = static_cast<T *>(t.buffer()->raw_data());
  size_t num_elements = total_size(t.shape());
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = dist(utils::RNG::instance().engine());
  }
}

/**
 * @brief Fill a parameter with values drawn from a normal distribution.
 * @tparam T Data type of the parameter.
 * @param t Parameter to fill.
 * @param mean Mean of the distribution.
 * @param stddev Standard deviation of the distribution.
 */
template <typename T> void normal(Parameter &t, T mean, T stddev) {
  std::normal_distribution<T> dist(mean, stddev);
  t.check_valid_dtype<T>();
  T *data = static_cast<T *>(t.buffer()->raw_data());
  size_t num_elements = total_size(t.shape());
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = dist(utils::RNG::instance().engine());
  }
}

/**
 * @brief Fill a parameter with values drawn from a uniform distribution
 * (float).
 * @param t Parameter to fill.
 * @param low Lower bound of the distribution.
 * @param high Upper bound of the distribution.
 */
void uniform(Parameter &t, float low, float high);

/**
 * @brief Fill a parameter with values drawn from a normal distribution (float).
 * @param t Parameter to fill.
 * @param mean Mean of the distribution.
 * @param stddev Standard deviation of the distribution.
 */
void normal(Parameter &t, float mean, float stddev);

/**
 * @brief Initialize a parameter using He uniform initialization.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with He uniform.
 */
Parameter he_uniform(const core::Shape &shape,
                     core::DType dtype = core::DType::FLOAT32,
                     core::Device device = core::Device::cpu());

/**
 * @brief Initialize a parameter using He normal initialization.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with He normal.
 */
Parameter he_normal(const core::Shape &shape,
                    core::DType dtype = core::DType::FLOAT32,
                    core::Device device = core::Device::cpu());

/**
 * @brief Initialize a parameter using Glorot (Xavier) uniform initialization.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with Glorot uniform.
 */
Parameter glorot_uniform(const core::Shape &shape,
                         core::DType dtype = core::DType::FLOAT32,
                         core::Device device = core::Device::cpu());

/**
 * @brief Initialize a parameter using Glorot (Xavier) normal initialization.
 * @param shape Shape of the parameter.
 * @param dtype Data type (default: FLOAT32).
 * @param device Device to allocate on (default: CPU).
 * @return Parameter initialized with Glorot normal.
 */
Parameter glorot_normal(const core::Shape &shape,
                        core::DType dtype = core::DType::FLOAT32,
                        core::Device device = core::Device::cpu());

/**
 * @brief Initialize a parameter using the specified initialization scheme.
 * @param shape Shape of the parameter.
 * @param dtype Data type.
 * @param device Device to allocate on.
 * @param init Initialization scheme to use.
 * @return Parameter initialized according to the scheme.
 */
Parameter initialize(const core::Shape &shape, core::DType dtype,
                     core::Device device, Initialization init);

} // namespace quasai::nn
