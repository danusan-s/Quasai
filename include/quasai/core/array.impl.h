/**
 * @file array.impl.h
 * @brief Implementation of Array class template
 *
 * This file provides the implementation for the Array class.
 * Include this file AFTER the class declaration in array.h
 *
 * Key implementations needed:
 * - Constructors and destructors
 * - Copy and move semantics
 * - Memory management (reserve, resize)
 * - Element access methods
 */

#include "array.h"
#include "error.h"

namespace quasai {

template <typename T>
Array<T>::Array() : data_(nullptr), size_(0), capacity_(0) {
}

template <typename T> Array<T>::Array(size_t size) {
  size_t new_capacity = next_power_of_two(size);
  data_ = new T[new_capacity];
  size_ = size;
  capacity_ = new_capacity;
}

template <typename T> Array<T>::~Array() {
  delete[] data_;
}

template <typename T>
Array<T>::Array(const Array &other)
    : data_(new T[other.capacity_]), size_(other.size_),
      capacity_(other.capacity_) {
  std::copy(other.data_, other.data_ + other.size_, data_);
}

template <typename T> Array<T>::Array(std::initializer_list<T> init) {
  size_t new_capacity = next_power_of_two(init.size());
  data_ = new T[new_capacity];
  size_ = init.size();
  capacity_ = new_capacity;
  std::copy(init.begin(), init.end(), data_);
}

template <typename T>
Array<T> &Array<T>::operator=(std::initializer_list<T> init) {
  if (init.size() > capacity_) {
    delete[] data_;
    data_ = new T[next_power_of_two(init.size())];
    capacity_ = next_power_of_two(init.size());
  }
  size_ = init.size();
  std::copy(init.begin(), init.end(), data_);
  return *this;
}

template <typename T> Array<T> &Array<T>::operator=(const Array &other) {
  if (this != &other) {
    delete[] data_;
    data_ = new T[other.capacity_];
    size_ = other.size_;
    capacity_ = other.capacity_;
    std::copy(other.data_, other.data_ + other.size_, data_);
  }
  return *this;
}

template <typename T>
Array<T>::Array(Array &&other) noexcept
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.capacity_ = 0;
}

template <typename T> Array<T> &Array<T>::operator=(Array &&other) noexcept {
  if (this != &other) {
    delete[] data_;
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }
  return *this;
}

template <typename T> void Array<T>::reserve(size_t new_capacity) {
  if (new_capacity > capacity_) {
    T *new_data = new T[new_capacity];
    std::copy(data_, data_ + size_, new_data);
    delete[] data_;
    data_ = new_data;
    capacity_ = new_capacity;
  }
}

template <typename T> void Array<T>::resize(size_t new_size) {
  if (new_size > capacity_) {
    reserve(new_size);
  }
  if (new_size > size_) {
    std::fill(data_ + size_, data_ + new_size, T());
  }
  size_ = new_size;
}

template <typename T> void Array<T>::clear() {
  size_ = 0;
}

template <typename T> void Array<T>::push_back(const T &value) {
  if (size_ >= capacity_) {
    reserve(capacity_ == 0 ? 1 : capacity_ * 2);
  }
  data_[size_++] = value;
}

template <typename T> T &Array<T>::operator[](size_t index) {
  if (index >= size_) {
    throw DimensionException("Index out of bounds");
  }
  return data_[index];
}

template <typename T> const T &Array<T>::operator[](size_t index) const {
  if (index >= size_) {
    throw DimensionException("Index out of bounds");
  }
  return data_[index];
}

template <typename T> size_t Array<T>::size() const {
  return size_;
}

template <typename T> size_t Array<T>::capacity() const {
  return capacity_;
}

template <typename T> T *Array<T>::begin() {
  return data_;
}

template <typename T> T *Array<T>::end() {
  return data_ + size_;
}

template <typename T> const T *Array<T>::begin() const {
  return data_;
}

template <typename T> const T *Array<T>::end() const {
  return data_ + size_;
}

} // namespace quasai
