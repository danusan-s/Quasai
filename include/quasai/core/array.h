#pragma once
/**
 * @file array.h
 * @brief Dynamic array implementation with contiguous memory
 *
 * A lightweight dynamic array that serves as the foundation for tensor storage.
 * Implement memory management, element access, and basic operations.
 *
 * Key features to implement:
 * - Dynamic resizing with capacity management
 * - RAII memory management
 * - Iterator support
 * - Element access with bounds checking
 * - Move semantics
 */

#include <cstddef>
#include <initializer_list>
#include <memory>
namespace quasai {

template <typename T> class Array {
public:
  Array();
  explicit Array(size_t size);
  ~Array();

  Array(std::initializer_list<T> init);
  Array &operator=(std::initializer_list<T> init);

  // copy and move semantics
  Array(const Array &other);
  Array &operator=(const Array &other);
  Array(Array &&other) noexcept;
  Array &operator=(Array &&other) noexcept;

  // Element access
  T &operator[](size_t index);
  const T &operator[](size_t index) const;

  // Capacity management
  size_t size() const;
  size_t capacity() const;
  void reserve(size_t new_capacity);
  void resize(size_t new_size);

  void clear();
  void push_back(const T &value);

  // Iterator support
  T *begin();
  T *end();
  const T *begin() const;
  const T *end() const;

private:
  T *data_;
  size_t size_;
  size_t capacity_;

  size_t next_power_of_two(size_t n) const {
    if (n == 0)
      return 1;
    if ((n & (n - 1)) == 0)
      return n; // already a power of two
    size_t power = 1;
    while (power < n) {
      power <<= 1;
    }
    return power;
  }
};

} // namespace quasai
