#include "quasai/storage/allocator.hpp"

namespace quasai {

/**
 * RAII wrapper for memory allocated by an Allocator. Automatically deallocates
 * memory when the Buffer goes out of scope. Similar to std::unique_ptr
 * Raw byte storage that can be used for any data type.
 */
class Buffer {
public:
  Buffer(Allocator *allocator = &CpuAllocator::instance(), std::size_t size = 0)
      : allocator_(allocator), size_(size) {
    data_ = allocator_->allocate(size_);
  }

  ~Buffer() {
    if (data_) {
      allocator_->deallocate(data_);
    }
    data_ = nullptr;
  }

  // Non-copyable
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  // Movable
  Buffer(Buffer &&other) noexcept
      : allocator_(other.allocator_), data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  Buffer &operator=(Buffer &&other) noexcept {
    if (this != &other) {
      // Deallocate current memory
      if (data_) {
        allocator_->deallocate(data_);
      }
      // Move data from other
      allocator_ = other.allocator_;
      data_ = other.data_;
      size_ = other.size_;
      // Reset other
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void *raw_data() const {
    return data_;
  }
  std::size_t size() const {
    return size_;
  }

private:
  Allocator *allocator_;
  void *data_;
  std::size_t size_;
};

} // namespace quasai
