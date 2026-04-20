#include <cstddef>
#include <initializer_list>
#include <stdexcept>

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

  std::size_t dimensions() const {
    return dimensions_;
  }

  std::size_t operator[](std::size_t index) const {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
  }

  std::size_t operator[](std::size_t index) {
    if (index >= dimensions_) {
      throw std::out_of_range("Dimension index out of range");
    }
    return sizes_[index];
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

} // namespace quasai
