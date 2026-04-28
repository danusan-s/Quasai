#pragma once

#include "quasai/storage/allocator.hpp"

namespace quasai::storage {

class Buffer {
public:
  Buffer(Allocator *allocator = &CpuAllocator::instance(),
         std::size_t size = 0);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  void *raw_data() const;
  std::size_t size() const;

private:
  Allocator *allocator_;
  void *data_;
  std::size_t size_;
};

} // namespace quasai::storage
