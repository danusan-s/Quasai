#pragma once

#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstring>
#include <memory>

namespace quasai {

struct TensorImpl {
  std::shared_ptr<Buffer> buffer;

  Shape shape;
  Strides strides;

  size_t offset;
  DType dtype;
  Device device;
};

class Tensor {
public:
  static Tensor zeros(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  static Tensor empty(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu());

  static Tensor from_data(const void *data, const Shape &shape, DType dtype,
                          Device device = Device::cpu());

  static Tensor from_impl(const TensorImpl &impl);

  static Allocator *allocator_for_device(const Device &device);

  TensorImpl get_impl() const;

  void reshape(const Shape &new_shape);

private:
  Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, DType dtype, Device device);

  TensorImpl impl_;
};

} // namespace quasai
