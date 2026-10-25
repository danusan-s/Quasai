#pragma once

#include "quasai/storage/allocator.hpp"
#include <memory>

namespace quasai::storage {

/**
 * @brief Buffer managing a block of memory, allocated via an Allocator.
 * @note Uses RAII — memory is automatically freed when the Buffer is destroyed.
 */
class Buffer {
public:
  struct Deleter {
    void operator()(void *ptr) const noexcept {
      if (ptr) {
        CpuAllocator::instance().deallocate(ptr);
      }
    }
  };

  /**
   * @brief Construct a Buffer with the given allocator and size.
   * @param allocator Allocator to use (default: CpuAllocator).
   * @param size Size in bytes to allocate (default: 0).
   */
  Buffer(Allocator *allocator = &CpuAllocator::instance(),
         std::size_t size = 0);
  ~Buffer() = default;

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept = default;
  Buffer &operator=(Buffer &&other) noexcept = default;

  /// @brief Get raw pointer to the allocated memory.
  void *raw_data() const {
    return data_.get();
  }
  /// @brief Get the size of the buffer in bytes.
  std::size_t size() const {
    return size_;
  }

private:
  std::unique_ptr<void, Deleter> data_;
  std::size_t size_;
};

} // namespace quasai::storage
