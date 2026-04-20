#include "quasai/storage/allocator.hpp"

namespace quasai {

void *CpuAllocator::allocate(std::size_t size) {
  if (size == 0) {
    size = 1;
  }
  return std::malloc(size);
}

void CpuAllocator::deallocate(void *ptr) {
  std::free(ptr);
}

CpuAllocator &CpuAllocator::instance() {
  static CpuAllocator allocator;
  return allocator;
}

} // namespace quasai
