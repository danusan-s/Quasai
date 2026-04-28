#include "quasai/storage/buffer.hpp"

namespace quasai::storage {

Buffer::Buffer(Allocator *allocator, std::size_t size) : size_(size) {
  void *allocated = allocator->allocate(size_);
  data_ = std::unique_ptr<void, Deleter>{
      allocated,
      Deleter{[allocator](void *ptr) { allocator->deallocate(ptr); }}};
}

} // namespace quasai::storage
