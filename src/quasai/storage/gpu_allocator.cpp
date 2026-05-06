#include "quasai/storage/allocator.hpp"

#ifdef QUASAI_CUDA
#include <cuda_runtime.h>
#endif

namespace quasai::storage {

#ifdef QUASAI_CUDA
void *CudaAllocator::allocate(std::size_t size) {
  if (size == 0) {
    size = 1;
  }
  void *ptr;
  cudaMalloc(&ptr, size);
  return ptr;
}

void CudaAllocator::deallocate(void *ptr) {
  cudaFree(ptr);
}

CudaAllocator &CudaAllocator::instance() {
  static CudaAllocator allocator;
  return allocator;
}
#endif

} // namespace quasai::storage
