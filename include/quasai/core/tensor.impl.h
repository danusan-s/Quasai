/**
 * @file tensor.impl.h
 * @brief Implementation of Tensor class template
 *
 * This file provides the implementation for the Tensor class.
 * Include this file AFTER the class declaration in tensor.h
 *
 * Key implementations needed:
 * - index_to_offset: Convert N-dimensional index to 1D offset (row-major)
 * - offset_to_index: Convert 1D offset back to N-dimensional index
 * - reshape: Change tensor shape while keeping data contiguous
 * - Element access with bounds checking
 */

#include "error.h"
#include "tensor.h"

namespace quasai {

template <typename T> Tensor<T>::Tensor() : data_(nullptr), shape_() {};

template <typename T>
Tensor<T>::Tensor(const Array<size_t> &shape) : data_(nullptr), shape_(shape) {
  size_ = 1;
  for (size_t dim : shape_) {
    size_ *= dim;
  }
  data_.resize(size_);
}

template <typename T> Tensor<T>::~Tensor() {
  // No manual memory management needed, Array will handle it
}

template <typename T>
Tensor<T>::Tensor(const Tensor &other)
    : data_(other.data_), shape_(other.shape_), size_(other.size_) {
}

template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor &other) {
  if (this != &other) {
    data_ = other.data_;
    shape_ = other.shape_;
    size_ = other.size_;
  }
  return *this;
}

template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept
    : data_(std::move(other.data_)), shape_(std::move(other.shape_)),
      size_(other.size_) {
  other.size_ = 0;
}

template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    data_ = std::move(other.data_);
    shape_ = std::move(other.shape_);
    size_ = other.size_;
    other.size_ = 0;
  }
  return *this;
}

template <typename T> T &Tensor<T>::operator()(const Array<size_t> &indices) {
  size_t index = 0;
  size_t stride = size_;
  for (size_t i = 0; i < shape_.size(); ++i) {
    stride /= shape_[i];
    index += indices[i] * stride;
  }
  return data_[index];
}

template <typename T>
const T &Tensor<T>::operator()(const Array<size_t> &indices) const {
  size_t index = 0;
  size_t stride = size_;
  for (size_t i = 0; i < shape_.size(); ++i) {
    stride /= shape_[i];
    index += indices[i] * stride;
  }
  return data_[index];
}

template <typename T> void Tensor<T>::reshape(const Array<size_t> &new_shape) {
  size_t new_size = 1;
  for (size_t dim : new_shape) {
    new_size *= dim;
  }
  if (new_size != size_) {
    throw InvalidParameterException(
        "New shape must have the same number of elements");
  }
  shape_ = new_shape;
}

// TODO: Figure out N-dimensional transpose logic
template <typename T> void Tensor<T>::transpose(const Array<size_t> &perm) {
}

template <typename T> void Tensor<T>::flatten() {
  shape_ = {size_};
}

} // namespace quasai
