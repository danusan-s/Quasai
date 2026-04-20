#include "quasai/core/device.hpp"
#include "quasai/core/dtype.hpp"
#include "quasai/core/shape.hpp"
#include "quasai/storage/buffer.hpp"
#include <cstring>
#include <memory>

namespace quasai {

struct TensorView {
  void *data; // Pointer to the tensor data

  Shape shape;     // Shape of the tensor
  Strides strides; // Strides for indexing

  size_t offset; // Offset in the buffer for this tensor
  DType dtype;   // Data type of the tensor
  Device device; // Device where the tensor is stored
};

class Tensor {
public:
  static Tensor zeros(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu()) {
    Buffer buffer = Buffer(allocator_for_device(device),
                           total_size(shape) * dtype_size(dtype));

    std::memset(buffer.raw_data(), 0, buffer.size());
    return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                  get_strides(shape), 0, dtype, device);
  }

  static Tensor empty(const Shape &shape, DType dtype = DType::FLOAT32,
                      Device device = Device::cpu()) {
    Buffer buffer = Buffer(allocator_for_device(device),
                           total_size(shape) * dtype_size(dtype));
    return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                  get_strides(shape), 0, dtype, device);
  }

  static Tensor from_data(const void *data, const Shape &shape, DType dtype,
                          Device device = Device::cpu()) {
    Buffer buffer = Buffer(allocator_for_device(device),
                           total_size(shape) * dtype_size(dtype));
    std::memcpy(buffer.raw_data(), data, buffer.size());
    return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                  get_strides(shape), 0, dtype, device);
  }

  static Allocator *allocator_for_device(const Device &device) {
    switch (device.type) {
      case DeviceType::CPU:
        return &CpuAllocator::instance();
      case DeviceType::GPU_CUDA:
        // Return GPU allocator instance here
        throw std::runtime_error("GPU allocator not implemented");
      default:
        throw std::runtime_error("Unsupported device type");
    }
  }

  TensorView view() const {
    return TensorView{
        buffer_->raw_data(), shape_, strides_, offset_, dtype_, device_};
  }

  void reshape(const Shape &new_shape) {
    if (total_size(new_shape) != total_size(shape_)) {
      throw std::runtime_error(
          "Total size must remain the same when reshaping");
    }
    shape_ = new_shape;
    strides_ = get_strides(new_shape);
  }

private:
  Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
         const Strides &strides, size_t offset, DType dtype, Device device)
      : buffer_(std::move(buffer)), shape_(shape), strides_(strides),
        offset_(offset), dtype_(dtype), device_(device) {
  }

  std::shared_ptr<Buffer> buffer_;
  Shape shape_;
  Strides strides_;
  size_t offset_;
  DType dtype_;
  Device device_;
};

} // namespace quasai
