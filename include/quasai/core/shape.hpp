#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace quasai {

constexpr std::size_t MAX_DIMENSIONS = 8;

class Shape {
public:
  Shape() : dimensions_(0) {
    for (std::size_t i = 0; i < MAX_DIMENSIONS; ++i) {
      sizes_[i] = 0;
    }
  }

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

  std::size_t dimensions() const {
    return dimensions_;
  }

  std::size_t operator[](std::size_t index) const {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
  }

  std::size_t &operator[](std::size_t index) {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
  }

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

  bool operator!=(const Shape &other) const {
    return !(*this == other);
  }

  auto begin() const {
    return sizes_;
  }

  auto end() const {
    return sizes_ + dimensions_;
  }

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

inline std::size_t total_size(const Shape &shape) {
  std::size_t size = 1;
  for (std::size_t i = 0; i < shape.dimensions(); ++i) {
    size *= shape[i];
  }
  return size;
}

using Strides = Shape;

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

inline size_t ravel_index(const Index &indices, const Shape &shape) {
  size_t flat_index = 0;
  size_t stride = 1;
  for (size_t i = shape.dimensions(); i > 0; --i) {
    flat_index += indices[i - 1] * stride;
    stride *= shape[i - 1];
  }
  return flat_index;
}

// Take in indices from the broadcasted tensor and return the mapped indices for
// the original tensor before applying broadcasting
// Simply take the first index (0) if the dimension is 1 as this means it was
// broadcasted up to the larger shape, otherwise just take the index as is
inline Index get_broadcast_index(const Index &indices, const Shape &shape) {
  std::size_t ndim = shape.dimensions();
  Index broadcast_index(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    broadcast_index[ndim - 1 - i] =
        shape[ndim - 1 - i] == 1 ? 0 : indices[indices.size() - 1 - i];
  }
  return broadcast_index;
}

inline Index unravel_index(size_t flat_index, const Shape &shape) {
  Index indices(shape.dimensions());
  for (size_t i = shape.dimensions(); i > 0; --i) {
    indices[i - 1] = flat_index % shape[i - 1];
    flat_index /= shape[i - 1];
  }
  return indices;
}

} // namespace quasai
