#include "quasai/storage/buffer.hpp"

namespace quasai {

Buffer::Buffer(Allocator *allocator, std::size_t size)
    : allocator_(allocator), size_(size) {
  data_ = allocator_->allocate(size_);
}

Buffer::~Buffer() {
  if (data_) {
    allocator_->deallocate(data_);
  }
  data_ = nullptr;
}

Buffer::Buffer(Buffer &&other) noexcept
    : allocator_(other.allocator_), data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    if (data_) {
      allocator_->deallocate(data_);
    }
    allocator_ = other.allocator_;
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void *Buffer::raw_data() const {
  return data_;
}

std::size_t Buffer::size() const {
  return size_;
}

} // namespace quasai
