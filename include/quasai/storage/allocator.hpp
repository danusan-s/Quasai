#pragma once

#include <cstddef>
#include <cstdlib>
#include <unordered_map>

namespace quasai::storage {

struct Block {
  void *ptr;
  size_t size;
  bool allocated;

  Block *prev; // physically adjacent
  Block *next;
};

/**
 * @brief Abstract base class for memory allocators.
 */
class Allocator {
public:
  /// @brief Allocate a block of memory.
  virtual void *allocate(std::size_t size) = 0;
  /// @brief Deallocate a block of memory.
  virtual void deallocate(void *ptr) = 0;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;
  virtual ~Allocator() = default;

protected:
  Allocator() = default;
};

/// @brief CPU memory allocator using std::malloc/std::free.
class CpuAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override;
  void deallocate(void *ptr) override;

  /// @brief Get the singleton instance.
  static CpuAllocator &instance();

  size_t align(std::size_t size);
  void split_block(Block *block, std::size_t size);

private:
  std::unordered_map<void *, Block *>
      allocations_;       // Map from ptr to block for deallocation
  Block *head_ = nullptr; // Start of the memory pool
  Block *tail_ = nullptr; // End of the memory pool
  CpuAllocator();
};

/// @brief CUDA GPU memory allocator using cudaMalloc/cudaFree.
class CudaAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override;
  void deallocate(void *ptr) override;

  /// @brief Get the singleton instance.
  static CudaAllocator &instance();
};

} // namespace quasai::storage
