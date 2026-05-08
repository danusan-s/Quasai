#include "quasai/core/tensor.hpp"

#include "quasai/autograd/engine.hpp"
#include "quasai/autograd/metadata.hpp"
#include "quasai/storage/allocator.hpp"

#include <memory>

namespace quasai::core {

using storage::Allocator;

Allocator *Tensor::allocator_for_device(const Device &device) {
  switch (device.type_) {
    case DeviceType::CPU:
      return &storage::CpuAllocator::instance();
    case DeviceType::GPU_CUDA:
      throw std::runtime_error("GPU allocator not implemented");
    default:
      throw std::runtime_error("Unsupported device type");
  }
}

Tensor Tensor::zeros(const Shape &shape, DType dtype, Device device) {
  storage::Buffer buffer = storage::Buffer(
      allocator_for_device(device), total_size(shape) * dtype_size(dtype));

  std::memset(buffer.raw_data(), 0, buffer.size());
  return Tensor(std::make_shared<storage::Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::ones(const Shape &shape, DType dtype, Device device) {
  storage::Buffer buffer = storage::Buffer(
      allocator_for_device(device), total_size(shape) * dtype_size(dtype));

  size_t count = total_size(shape);

  switch (dtype) {
    case DType::FLOAT32: {
      float *data = static_cast<float *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0f;
      }
      break;
    }
    case DType::FLOAT64: {
      double *data = static_cast<double *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0;
      }
      break;
    }
    case DType::INT32: {
      int32_t *data = static_cast<int32_t *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1;
      }
      break;
    }
    case DType::INT64: {
      int64_t *data = static_cast<int64_t *>(buffer.raw_data());
      for (size_t i = 0; i < count; ++i) {
        data[i] = 1;
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for ones");
  }

  return Tensor(std::make_shared<storage::Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device) {
  storage::Buffer buffer = storage::Buffer(
      allocator_for_device(device), total_size(shape) * dtype_size(dtype));
  return Tensor(std::make_shared<storage::Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::from_data(const void *data, const Shape &shape, DType dtype,
                         Device device) {
  storage::Buffer buffer = storage::Buffer(
      allocator_for_device(device), total_size(shape) * dtype_size(dtype));
  std::memcpy(buffer.raw_data(), data, buffer.size());
  return Tensor(std::make_shared<storage::Buffer>(std::move(buffer)), shape,
                get_strides(shape), 0, true, dtype, device);
}

Tensor Tensor::from_impl(const TensorImpl &impl) {
  return Tensor(impl.buffer_, impl.shape_, impl.strides_, impl.offset_,
                impl.is_contiguous_, impl.dtype_, impl.device_);
}

std::shared_ptr<storage::Buffer> Tensor::buffer() const {
  return impl_.buffer_;
}

const Shape &Tensor::shape() const {
  return impl_.shape_;
}

const Strides &Tensor::strides() const {
  return impl_.strides_;
}

bool Tensor::is_contiguous() const {
  return impl_.is_contiguous_;
}

DType Tensor::dtype() const {
  return impl_.dtype_;
}

Device Tensor::device() const {
  return impl_.device_;
}

std::shared_ptr<autograd::AutoGradMeta> Tensor::autograd_meta() const {
  return impl_.autograd_meta_;
}

void Tensor::requires_grad(bool grad_needed) {
  if (!impl_.autograd_meta_) {
    impl_.autograd_meta_ = std::make_shared<autograd::AutoGradMeta>();
  }
  impl_.autograd_meta_->requires_grad_ = grad_needed;
}

void Tensor::set_grad_fn(std::unique_ptr<autograd::Function> grad_fn) {
  if (!impl_.autograd_meta_) {
    impl_.autograd_meta_ = std::make_shared<autograd::AutoGradMeta>();
  }
  impl_.autograd_meta_->grad_fn_ = std::move(grad_fn);
}

bool Tensor::is_valid() const {
  return impl_.buffer_ != nullptr;
}

TensorImpl Tensor::get_impl_copy() const {
  TensorImpl copy = impl_;
  copy.autograd_meta_ = nullptr;
  return copy;
}

void Tensor::backward() {
  autograd::AutoGradEngine::backward(*this);
}

Tensor::Tensor()
    : impl_(TensorImpl{nullptr, Shape(), Strides(), 0, true, DType::FLOAT32,
                       Device::cpu(), nullptr}) {
}

Tensor::Tensor(std::shared_ptr<storage::Buffer> buffer, const Shape &shape,
               const Strides &strides, size_t offset, bool is_contiguous,
               DType dtype, Device device)
    : impl_(TensorImpl{std::move(buffer), shape, strides, offset, is_contiguous,
                       dtype, device, nullptr}) {
}

} // namespace quasai::core
