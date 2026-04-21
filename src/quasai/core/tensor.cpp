#include "quasai/core/tensor.hpp"

namespace quasai {

Allocator *Tensor::allocator_for_device(const Device &device) {
  switch (device.type) {
    case DeviceType::CPU:
      return &CpuAllocator::instance();
    case DeviceType::GPU_CUDA:
      throw std::runtime_error("GPU allocator not implemented");
    default:
      throw std::runtime_error("Unsupported device type");
  }
}

Tensor Tensor::zeros(const Shape &shape, DType dtype, Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));

  std::memset(buffer.raw_data(), 0, buffer.size());
  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, dtype, device);
}

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));
  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, dtype, device);
}

Tensor Tensor::from_data(const void *data, const Shape &shape, DType dtype,
                         Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));
  std::memcpy(buffer.raw_data(), data, buffer.size());
  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, dtype, device);
}

// Same underlying buffer used by new tensor to create cheap copies
Tensor Tensor::from_impl(const TensorImpl &impl) {
  return Tensor(impl.buffer, impl.shape, impl.strides, impl.offset, impl.dtype,
                impl.device);
}

TensorImpl Tensor::get_impl() const {
  return impl_;
}

void Tensor::reshape(const Shape &new_shape) {
  if (total_size(new_shape) != total_size(impl_.shape)) {
    throw std::runtime_error("Total size must remain the same when reshaping");
  }
  impl_.shape = new_shape;
  impl_.strides = get_strides(new_shape);
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
               const Strides &strides, size_t offset, DType dtype,
               Device device)
    : impl_{std::move(buffer), shape, strides, offset, dtype, device} {
}

} // namespace quasai
