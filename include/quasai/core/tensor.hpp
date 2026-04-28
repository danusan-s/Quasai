#pragma once

#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace quasai::core {

class Function;
class AutoGradMeta;

struct TensorImpl {
  std::shared_ptr<Buffer> buffer;

  Shape shape;
  Strides strides;
  size_t offset;
  bool is_contiguous;

  DType dtype;
  Device device;

  std::shared_ptr<AutoGradMeta> autograd_meta;
};

class Tensor {
public:
  static Tensor zeros(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  static Tensor ones(const Shape &shape, DType dtype = DType::FLOAT32,
                     Device device = Device::cpu());

  static Tensor empty(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  static Tensor from_data(const void *data, const Shape &shape, DType dtype,
                          Device device = Device::cpu());

  static Tensor from_impl(const TensorImpl &impl);

  template <typename T>
  static Tensor from_scalar(T scalar, DType dtype = DTypeTraits<T>::dtype,
                            Device device = Device::cpu()) {

    Tensor out = Tensor::empty({}, dtype, device);

    switch (dtype) {
      case DType::INT32:
        *out.data<int32_t>() = static_cast<int32_t>(scalar);
        break;
      case DType::INT64:
        *out.data<int64_t>() = static_cast<int64_t>(scalar);
        break;
      case DType::FLOAT32:
        *out.data<float>() = static_cast<float>(scalar);
        break;
      case DType::FLOAT64:
        *out.data<double>() = static_cast<double>(scalar);
        break;
      default:
        throw std::runtime_error("Unsupported dtype");
    }

    return out;
  }

  static Allocator *allocator_for_device(const Device &device);

  template <typename T> void check_valid_dtype() const {
    if (impl_.dtype != DTypeTraits<T>::dtype) {
      throw std::runtime_error(
          "Requested data type does not match tensor dtype");
    }
  }

  template <typename T> T *data() {
    check_valid_dtype<T>();
    return static_cast<T *>(impl_.buffer->raw_data()) + impl_.offset;
  }

  template <typename T> const T *data() const {
    check_valid_dtype<T>();
    return static_cast<const T *>(impl_.buffer->raw_data()) + impl_.offset;
  }

  template <typename T> T &at(Index index) {
    check_valid_dtype<T>();
    size_t flat = ravel_index(index, impl_.strides);
    return data<T>()[flat];
  }

  std::shared_ptr<Buffer> buffer() const;
  const Shape &shape() const;
  const Strides &strides() const;
  bool is_contiguous() const;
  DType dtype() const;
  Device device() const;
  std::shared_ptr<AutoGradMeta> autograd_meta() const;

  void requires_grad(bool grad_needed);
  void set_grad_fn(std::unique_ptr<Function> grad_fn);

  TensorImpl get_impl_copy() const;

  void backward();

  Tensor();

private:
  Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, bool is_contiguous, DType dtype,
         Device device);

  TensorImpl impl_;
};

} // namespace quasai::core
