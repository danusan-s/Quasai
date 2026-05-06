#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace quasai::core {

/// @brief Maximum number of dimensions supported by a tensor.
constexpr std::size_t MAX_DIMENSIONS = 8;

/**
 * @brief Represents the shape (dimensions) of a tensor.
 * @note Maximum dimensions is capped at MAX_DIMENSIONS (8).
 */
class Shape {
public:
  /// @brief Construct an empty shape (0 dimensions).
  Shape() : dimensions_(0) {
    for (std::size_t i = 0; i < MAX_DIMENSIONS; ++i) {
      sizes_[i] = 0;
    }
  }

  /**
   * @brief Construct a shape from an initializer list.
   * @param dims Initializer list of dimension sizes.
   * @throws std::runtime_error if dims exceed MAX_DIMENSIONS.
   */
  Shape(std::initializer_list<size_t> dims) : dimensions_(dims.size()) {
    size_t i = 0;
    for (size_t dim : dims) {
      if (i < MAX_DIMENSIONS) {
        sizes_[i++] = dim;
      } else {
        throw std::runtime_error("Exceeded maximum dimensions");
      }
    }
    for (; i < MAX_DIMENSIONS; ++i) {
      sizes_[i] = 0;
    }
  }

  /**
   * @brief Construct a shape from a raw array of dimension sizes.
   * @param dims Pointer to array of dimension sizes.
   * @param num_dims Number of dimensions.
   * @throws std::runtime_error if num_dims exceeds MAX_DIMENSIONS.
   */
  Shape(const std::size_t *dims, std::size_t num_dims) : dimensions_(num_dims) {
    if (num_dims > MAX_DIMENSIONS) {
      throw std::runtime_error("Exceeded maximum dimensions");
    }
    for (std::size_t i = 0; i < num_dims; ++i) {
      sizes_[i] = dims[i];
    }
    for (std::size_t i = num_dims; i < MAX_DIMENSIONS; ++i) {
      sizes_[i] = 0;
    }
  }

  /**
   * @brief Construct a shape with n_dims dimensions, each initialized to 1.
   * @param n_dims Number of dimensions.
   * @throws std::runtime_error if n_dims exceeds MAX_DIMENSIONS.
   */
  Shape(std::size_t n_dims) : dimensions_(n_dims) {
    if (n_dims > MAX_DIMENSIONS) {
      throw std::runtime_error("Exceeded maximum dimensions");
    }
    for (std::size_t i = 0; i < n_dims; ++i) {
      sizes_[i] = 1;
    }
    for (std::size_t i = n_dims; i < MAX_DIMENSIONS; ++i) {
      sizes_[i] = 0;
    }
  }

  /// @brief Get the number of dimensions.
  std::size_t dimensions() const {
    return dimensions_;
  }

  /**
   * @brief Get dimension size (const version).
   * @param index Dimension index (0-based).
   * @return Size of the dimension.
   * @throws std::out_of_range if index is out of range.
   */
  std::size_t operator[](std::size_t index) const {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
  }

  /**
   * @brief Get dimension size (mutable version).
   * @param index Dimension index (0-based).
   * @return Reference to the dimension size.
   * @throws std::out_of_range if index is out of range.
   */
  std::size_t &operator[](std::size_t index) {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
  }

  /// @brief Compare two shapes for equality.
  bool operator==(const Shape &other) const {
    if (dimensions_ != other.dimensions_) {
      return false;
    }
    for (std::size_t i = 0; i < dimensions_; ++i) {
      if (sizes_[i] != other.sizes_[i]) {
        return false;
      }
    }
    return true;
  }

  /// @brief Compare two shapes for inequality.
  bool operator!=(const Shape &other) const {
    return !(*this == other);
  }

  /// @brief Get iterator to beginning.
  auto begin() const {
    return sizes_;
  }

  /// @brief Get iterator to end.
  auto end() const {
    return sizes_ + dimensions_;
  }

  /// @brief Convert shape to string representation (e.g., "(2, 3)").
  std::string to_string() const {
    std::string result = "(";
    for (std::size_t i = 0; i < dimensions_; ++i) {
      result += std::to_string(sizes_[i]);
      if (i < dimensions_ - 1) {
        result += ", ";
      }
    }
    result += ")";
    return result;
  }

private:
  std::size_t dimensions_;
  std::size_t sizes_[MAX_DIMENSIONS];
};

/**
 * @brief Remove a dimension of size 1 from the shape.
 * @param shape Input shape.
 * @param dim Dimension to squeeze (must have size 1).
 * @return New shape with the dimension removed.
 * @throws std::out_of_range if dim is out of range.
 * @throws std::runtime_error if dimension size is not 1.
 */
inline Shape squeeze_shape(const Shape &shape, std::size_t dim) {
  if (dim >= shape.dimensions()) {
    throw std::out_of_range("Dimension index out of range for squeeze");
  }
  if (shape[dim] != 1) {
    throw std::runtime_error("Cannot squeeze dimension " + std::to_string(dim) +
                             " with size " + std::to_string(shape[dim]));
  }

  std::size_t new_dims[MAX_DIMENSIONS] = {0};
  std::size_t new_dim_count = 0;
  for (std::size_t i = 0; i < shape.dimensions(); ++i) {
    if (i != dim) {
      new_dims[new_dim_count++] = shape[i];
    }
  }
  return Shape(new_dims, new_dim_count);
}

/**
 * @brief Compute the total number of elements in a shape.
 * @param shape Input shape.
 * @return Product of all dimension sizes.
 */
inline std::size_t total_size(const Shape &shape) {
  std::size_t size = 1;
  for (std::size_t i = 0; i < shape.dimensions(); ++i) {
    size *= shape[i];
  }
  return size;
}

/// @brief Strides are stored as Shape (same representation).
using Strides = Shape;

/**
 * @brief Compute row-major strides for a shape.
 * @param shape Input shape.
 * @return Strides where strides[i] = product of shape[i+1:].
 */
inline Strides get_strides(const Shape &shape) {
  std::size_t strides[MAX_DIMENSIONS] = {0};
  std::size_t dim = shape.dimensions();
  if (dim == 0) {
    return Strides();
  }
  for (std::size_t i = dim; i > 0; --i) {
    strides[i - 1] = (i == dim) ? 1 : strides[i] * shape[i];
  }
  return Strides(strides, dim);
}

/**
 * @brief Compute the broadcasted shape of two tensors.
 * @param a Shape of first tensor.
 * @param b Shape of second tensor.
 * @return Broadcasted shape following NumPy rules.
 * @throws std::runtime_error if shapes are not broadcast-compatible.
 */
inline Shape broadcast_shape(const Shape &a, const Shape &b) {
  std::size_t max_dims = std::max(a.dimensions(), b.dimensions());
  std::size_t result_dims[MAX_DIMENSIONS] = {0};

  for (std::size_t i = 0; i < max_dims; ++i) {
    std::size_t dim_a = (i < a.dimensions()) ? a[a.dimensions() - 1 - i] : 1;
    std::size_t dim_b = (i < b.dimensions()) ? b[b.dimensions() - 1 - i] : 1;

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      throw std::runtime_error("Shapes " + a.to_string() + " and " +
                               b.to_string() +
                               " are not compatible for broadcasting");
    }
    result_dims[max_dims - 1 - i] = std::max(dim_a, dim_b);
  }
  return Shape(result_dims, max_dims);
}

using Index = std::vector<std::size_t>;

/**
 * @brief Convert a multi-dimensional index to a flat index.
 * @param indices Multi-dimensional index.
 * @param strides Strides for each dimension.
 * @return Flat (1D) index.
 */
inline size_t ravel_index(const Index &indices, const Strides &strides) {
  size_t flat_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    flat_index += indices[i] * strides[i];
  }
  return flat_index;
}

/**
 * @brief Map broadcasted indices back to original tensor indices.
 * @param indices Indices from the broadcasted tensor.
 * @param shape Shape of the original (pre-broadcast) tensor.
 * @return Indices mapped to the original tensor.
 * @note Dimensions of size 1 (broadcasted) map to index 0.
 */
inline Index get_broadcast_index(const Index &indices, const Shape &shape) {
  std::size_t ndim = shape.dimensions();
  Index broadcast_index(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    broadcast_index[ndim - 1 - i] =
        shape[ndim - 1 - i] == 1 ? 0 : indices[indices.size() - 1 - i];
  }
  return broadcast_index;
}

/**
 * @brief Convert a flat index back to a multi-dimensional index.
 * @param flat_idx Flat (1D) index.
 * @param shape Shape of the tensor.
 * @return Multi-dimensional index.
 */
inline Index unravel_index(size_t flat_idx, const Shape &shape) {
  Index idx(shape.dimensions());

  for (int i = shape.dimensions() - 1; i >= 0; --i) {
    idx[i] = flat_idx % shape[i];
    flat_idx /= shape[i];
  }

  return idx;
}

} // namespace quasai::core
