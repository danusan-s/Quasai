#pragma once

#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstring>
#include <memory>

namespace quasai {

class Function;
class AutoGradMeta;

struct TensorImpl {
  std::shared_ptr<Buffer> buffer;

  Shape shape;
  Strides strides;

  size_t offset;
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

  static Allocator *allocator_for_device(const Device &device);

  void reshape(const Shape &new_shape);

  template <typename T> void check_valid_dtype() const {
    if (impl_.dtype != DTypeTraits<T>::dtype) {
      throw std::runtime_error(
          "Requested data type does not match tensor dtype");
    }
  }

  template <typename T> T *data() {
    check_valid_dtype<T>();
    return static_cast<T *>(impl_.buffer->raw_data());
  }

  template <typename T> const T *data() const {
    check_valid_dtype<T>();
    return static_cast<const T *>(impl_.buffer->raw_data());
  }

  template <typename T> T at(Shape index) const {
    check_valid_dtype<T>();
    if (index.dimensions() > impl_.shape.dimensions()) {
      throw std::runtime_error(
          "Index dimensions greater than tensor dimensions");
    }
    size_t flat_index = impl_.offset;
    for (size_t i = 0; i < index.dimensions(); ++i) {
      if (index[i] >= impl_.shape[i]) {
        throw std::runtime_error("Index out of bounds");
      }
      flat_index += index[i] * impl_.strides[i];
    }
    return data<T>()[flat_index];
  }

  std::shared_ptr<Buffer> buffer() const;
  const Shape &shape() const;
  const Strides &strides() const;
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
         const Strides &strides, size_t offset, DType dtype, Device device);

  TensorImpl impl_;
};

} // namespace quasai
