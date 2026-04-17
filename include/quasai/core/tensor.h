#pragma once
/**
 * @file tensor.h
 * @brief N-dimensional tensor class for Quasai framework
 *
 * The Tensor class is the core data structure of the framework.
 * It stores multi-dimensional data and provides operations for
 * mathematical computations, reshaping, and memory management.
 *
 * Implement the following:
 * - Multi-dimensional shape storage
 * - Data access methods (row-major order)
 * - Basic operations (reshape, transpose, etc.)
 * - Element-wise operations
 * - Memory layout management
 */

#include "array.h"
namespace quasai {

template <typename T> class Tensor {
public:
  // Constructors and destructors
  Tensor();
  Tensor(const Array<size_t> &shape);
  ~Tensor();

  // Copy and move semantics
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  // Data access methods
  T &operator()(const Array<size_t> &indices);
  const T &operator()(const Array<size_t> &indices) const;

  // Basic operations
  void reshape(const Array<size_t> &new_shape);
  void transpose(const Array<size_t> &perm);
  void flatten();

private:
  Array<T> data_;
  Array<size_t> shape_;
  size_t size_;
};

} // namespace quasai
