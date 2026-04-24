#include "quasai/core/tensor.hpp"
#include "quasai/autograd/engine.hpp"
#include "quasai/autograd/metadata.hpp"
#include <memory>

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
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::ones(const Shape &shape, DType dtype, Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));

  size_t count = total_size(shape);

  // Fill with ones based on dtype
  switch (dtype) {
    case DType::FLOAT32:
    case DType::FLOAT64: {
      float *data = static_cast<float *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0f;
      }
      break;
    }
    case DType::INT32:
    case DType::INT64: {
      int *data = static_cast<int *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1;
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for ones");
  }

  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));
  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::from_data(const void *data, const Shape &shape, DType dtype,
                         Device device) {
  Buffer buffer = Buffer(allocator_for_device(device),
                         total_size(shape) * dtype_size(dtype));
  std::memcpy(buffer.raw_data(), data, buffer.size());
  return Tensor(std::make_shared<Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

// Same underlying buffer used by new tensor to create cheap copies
Tensor Tensor::from_impl(const TensorImpl &impl) {
  return Tensor(impl.buffer, impl.shape, impl.strides, impl.offset,
                impl.is_contiguous, impl.dtype, impl.device);
}

std::shared_ptr<Buffer> Tensor::buffer() const {
  return impl_.buffer;
}

const Shape &Tensor::shape() const {
  return impl_.shape;
}

const Strides &Tensor::strides() const {
  return impl_.strides;
}

bool Tensor::is_contiguous() const {
  return impl_.is_contiguous;
}

DType Tensor::dtype() const {
  return impl_.dtype;
}

Device Tensor::device() const {
  return impl_.device;
}

std::shared_ptr<AutoGradMeta> Tensor::autograd_meta() const {
  return impl_.autograd_meta;
}

void Tensor::requires_grad(bool grad_needed) {
  if (!impl_.autograd_meta) {
    impl_.autograd_meta = std::make_shared<AutoGradMeta>();
  }
  impl_.autograd_meta->requires_grad = grad_needed;
}

void Tensor::set_grad_fn(std::unique_ptr<Function> grad_fn) {
  if (!impl_.autograd_meta) {
    impl_.autograd_meta = std::make_shared<AutoGradMeta>();
  }
  impl_.autograd_meta->grad_fn = std::move(grad_fn);
}

TensorImpl Tensor::get_impl_copy() const {
  return impl_;
}

void Tensor::backward() {
  AutoGradEngine::backward(*this, Tensor());
}

Tensor::Tensor()
    : impl_(TensorImpl{nullptr, Shape(), Strides(), 0, true, DType::FLOAT32,
                       Device::cpu(), nullptr}) {
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, const Shape &shape,
               const Strides &strides, size_t offset, bool is_contiguous,
               DType dtype, Device device)
    : impl_(TensorImpl{std::move(buffer), shape, strides, offset, is_contiguous,
                       dtype, device, nullptr}) {
}

} // namespace quasai
