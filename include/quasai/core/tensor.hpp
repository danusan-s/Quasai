#pragma once

#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace quasai::autograd {
class AutoGradMeta;
class Function;
} // namespace quasai::autograd

namespace quasai::core {

/**
 * @brief Internal implementation detail for Tensor.
 * @note This is not part of the public API. Use Tensor's public methods
 *       (shape(), dtype(), etc.) instead of accessing this struct directly.
 */
struct TensorImpl {
  std::shared_ptr<storage::Buffer> buffer;

  Shape shape;
  Strides strides;
  size_t offset;
  bool is_contiguous;

  DType dtype;
  Device device;

  std::shared_ptr<autograd::AutoGradMeta> autograd_meta;
};

class Tensor {
public:
  /**
   * @brief Create a tensor filled with zeros.
   * @param shape Shape of the tensor.
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   * @return Tensor filled with zeros.
   */
  static Tensor zeros(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  /**
   * @brief Create a tensor filled with ones.
   * @param shape Shape of the tensor.
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   * @return Tensor filled with ones.
   */
  static Tensor ones(const Shape &shape, DType dtype = DType::FLOAT32,
                     Device device = Device::cpu());

  /**
   * @brief Create an uninitialized tensor with the given shape.
   * @param shape Shape of the tensor.
   * @param dtype Data type (default: FLOAT32).
   * @param device Device to allocate on (default: CPU).
   * @return Uninitialized tensor.
   */
  static Tensor empty(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  /**
   * @brief Create a tensor from existing data.
   * @param data Pointer to source data.
   * @param shape Shape of the tensor.
   * @param dtype Data type of the source data.
   * @param device Device to allocate on (default: CPU).
   * @return Tensor initialized with the given data.
   */
  static Tensor from_data(const void *data, const Shape &shape, DType dtype,
                          Device device = Device::cpu());

  /**
   * @brief Create a tensor from an existing TensorImpl.
   * @param impl TensorImpl to copy from (autograd_meta will be stripped).
   * @return New tensor sharing the same buffer.
   */
  static Tensor from_impl(const TensorImpl &impl);

  /**
   * @brief Create a scalar tensor from a single value.
   * @tparam T Type of the scalar (deduced from scalar).
   * @param scalar Value to store in the tensor.
   * @param dtype Data type (default: deduced from T).
   * @param device Device to allocate on (default: CPU).
   * @return Scalar tensor containing the value.
   */
  template <typename T>
  static Tensor from_scalar(T scalar, DType dtype = DTypeTraits<T>::dtype,
                            Device device = Device::cpu()) {

    Tensor out = Tensor::empty({}, dtype, device);

    switch (dtype) {
      case DType::FLOAT32: {
        float *data_ptr = out.data<float>();
        data_ptr[0] = static_cast<float>(scalar);
        break;
      }
      case DType::INT32: {
        int32_t *data_ptr = out.data<int32_t>();
        data_ptr[0] = static_cast<int32_t>(scalar);
        break;
      }
      case DType::FLOAT64: {
        double *data_ptr = out.data<double>();
        data_ptr[0] = static_cast<double>(scalar);
        break;
      }
      case DType::INT64: {
        int64_t *data_ptr = out.data<int64_t>();
        data_ptr[0] = static_cast<int64_t>(scalar);
        break;
      }
    }

    return out;
  }

  /**
   * @brief Get the allocator for a given device.
   * @param device Device to get allocator for.
   * @return Pointer to the appropriate Allocator.
   * @throws std::runtime_error if device type is not supported.
   */
  static storage::Allocator *allocator_for_device(const Device &device);

  template <typename T> void check_valid_dtype() const {
    if (impl_.dtype != DTypeTraits<T>::dtype) {
      throw std::runtime_error(
          "Requested data type does not match tensor dtype");
    }
  }

  /**
   * @brief Get mutable pointer to tensor data.
   * @tparam T Expected data type (must match tensor's dtype).
   * @return Pointer to the first element, adjusted for offset.
   * @throws std::runtime_error if T does not match tensor dtype.
   */
  template <typename T> T *data() {
    check_valid_dtype<T>();
    return static_cast<T *>(impl_.buffer->raw_data()) + impl_.offset;
  }

  /**
   * @brief Get const pointer to tensor data.
   * @tparam T Expected data type (must match tensor's dtype).
   * @return Const pointer to the first element, adjusted for offset.
   * @throws std::runtime_error if T does not match tensor dtype.
   */
  template <typename T> const T *data() const {
    check_valid_dtype<T>();
    return static_cast<const T *>(impl_.buffer->raw_data()) + impl_.offset;
  }

  /**
   * @brief Get reference to element at the given index.
   * @tparam T Expected data type (must match tensor's dtype).
   * @param index Multi-dimensional index (e.g., {0, 1} for 2D tensor).
   * @return Reference to the element.
   * @throws std::runtime_error if T does not match tensor dtype.
   */
  template <typename T> T &at(Index index) {
    check_valid_dtype<T>();
    size_t flat = ravel_index(index, impl_.strides);
    return data<T>()[flat];
  }

  /// @brief Get the underlying buffer.
  std::shared_ptr<storage::Buffer> buffer() const;
  /// @brief Get the shape of the tensor.
  const Shape &shape() const;
  /// @brief Get the strides of the tensor.
  const Strides &strides() const;
  /// @brief Check if the tensor is contiguous in memory.
  bool is_contiguous() const;
  /// @brief Get the data type of the tensor.
  DType dtype() const;
  /// @brief Get the device the tensor is allocated on.
  Device device() const;
  /// @brief Get the autograd metadata (may be null).
  std::shared_ptr<autograd::AutoGradMeta> autograd_meta() const;

  /**
   * @brief Enable or disable gradient computation for this tensor.
   * @param grad_needed true to track gradients, false to stop.
   */
  void requires_grad(bool grad_needed);
  /**
   * @brief Set the gradient function for backward pass.
   * @param grad_fn Unique pointer to the Function that produced this tensor.
   */
  void set_grad_fn(std::unique_ptr<autograd::Function> grad_fn);

  /**
   * @brief Returns a copy of the TensorImpl with autograd_meta stripped.
   * @note Useful for inspecting shape/strides/dtype without sharing
   *       gradient state.
   */
  TensorImpl get_impl_copy() const;

  /**
   * @brief Trigger backward pass starting from this tensor.
   * @note Initializes gradient to ones and traverses the compute graph.
   */
  void backward();

  /// @brief Check if the tensor is valid (has a non-null buffer).
  bool is_valid() const;

  /// @brief Default constructor. Creates an invalid tensor.
  Tensor();

private:
  Tensor(std::shared_ptr<storage::Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, bool is_contiguous, DType dtype,
         Device device);

  TensorImpl impl_;
};

} // namespace quasai::core
