#pragma once

#include "quasai/storage/allocator.hpp"
#include <functional>
#include <memory>

namespace quasai::storage {

class Buffer {
public:
  Buffer(Allocator *allocator = &CpuAllocator::instance(),
         std::size_t size = 0);
  ~Buffer() = default; // RAII: unique_ptr handles cleanup automatically

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept = default;
  Buffer &operator=(Buffer &&other) noexcept = default;

  void *raw_data() const {
    return data_.get();
  }
  std::size_t size() const {
    return size_;
  }

private:
  struct Deleter {
    std::function<void(void *)> deallocate;

    void operator()(void *ptr) const {
      if (ptr && deallocate)
        deallocate(ptr);
    }
  };

  std::unique_ptr<void, Deleter> data_;
  std::size_t size_;
};

} // namespace quasai::storage
