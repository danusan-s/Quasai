#include <cstddef>
#include <cstdlib>

namespace quasai {

class Allocator {
public:
  virtual void *allocate(std::size_t size) = 0;
  virtual void deallocate(void *ptr) = 0;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;
  ~Allocator() = default;

protected:
  Allocator() = default;
};

class CpuAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override {
    if (size == 0) {
      size = 1; // malloc(0) behavior is implementation-defined, so we allocate
                // at least 1 byte
    }
    return std::malloc(size);
  }
  void deallocate(void *ptr) override {
    std::free(ptr);
  }

  static CpuAllocator &instance() {
    static CpuAllocator allocator;
    return allocator;
  }
};

class GpuAllocator : public Allocator {
  // TODO: Implement GPU memory allocation and deallocation
};

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
