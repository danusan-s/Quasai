#pragma once

#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstring>
#include <memory>

namespace quasai {

struct TensorView {
  void *data;

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

  static Allocator *allocator_for_device(const Device &device);

  TensorView view() const;

  void reshape(const Shape &new_shape);

private:
  Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, DType dtype, Device device);

  std::shared_ptr<Buffer> buffer_;
  Shape shape_;
  Strides strides_;
  size_t offset_;
  DType dtype_;
  Device device_;
};

} // namespace quasai
