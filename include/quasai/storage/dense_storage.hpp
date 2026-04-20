#include "quasai/core/dtype.hpp"
#include "quasai/core/memory.hpp"
#include <cstring>
#include <memory>

namespace quasai {

/**
 * Wrapper around Buffer that includes count and dtype.
 * Provides shared ownership of Buffer.
 * Provides factory methods for different initialization patterns
 * (zeros, uninitialized, from_data)
 */
class DenseStorage {
public:
  static DenseStorage zeros(std::size_t count, DType dtype,
                            Allocator *allocator = &CpuAllocator::instance()) {
    DenseStorage storage = uninitialized(count, dtype, allocator);
    std::memset(storage.buffer()->raw_data(), 0, storage.buffer()->size());
    return storage;
  }

  static DenseStorage
  uninitialized(std::size_t count, DType dtype,
                Allocator *allocator = &CpuAllocator::instance()) {
    std::size_t byte_size = count * dtype_size(dtype);
    auto buffer = std::make_shared<Buffer>(allocator, byte_size);
    return DenseStorage(std::move(buffer), count, dtype);
  }

  template <typename T>
  static DenseStorage
  from_data(const T *data, std::size_t count, DType dtype,
            Allocator *allocator = &CpuAllocator::instance()) {
    DenseStorage storage = uninitialized(count, dtype, allocator);
    std::memcpy(storage.buffer()->raw_data(), data, storage.buffer()->size());
    return storage;
  }

  template <typename T> const T *data() const {
    return static_cast<const T *>(buffer_->raw_data());
  }

  std::shared_ptr<Buffer> buffer() const {
    return buffer_;
  }
  std::size_t count() const {
    return count_;
  }
  DType dtype() const {
    return dtype_;
  }

private:
  DenseStorage(std::shared_ptr<Buffer> buffer, std::size_t count, DType dtype)
      : buffer_(std::move(buffer)), count_(count), dtype_(dtype) {
  }

  std::shared_ptr<Buffer> buffer_;
  std::size_t count_;
  DType dtype_;
};

} // namespace quasai
