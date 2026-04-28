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

  static storage::Allocator *allocator_for_device(const Device &device);

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

  std::shared_ptr<storage::Buffer> buffer() const;
  const Shape &shape() const;
  const Strides &strides() const;
  bool is_contiguous() const;
  DType dtype() const;
  Device device() const;
  std::shared_ptr<autograd::AutoGradMeta> autograd_meta() const;

  void requires_grad(bool grad_needed);
  void set_grad_fn(std::unique_ptr<autograd::Function> grad_fn);

  TensorImpl get_impl_copy() const;

  void backward();

  Tensor();

private:
  Tensor(std::shared_ptr<storage::Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, bool is_contiguous, DType dtype,
         Device device);

  TensorImpl impl_;
};

} // namespace quasai::core
