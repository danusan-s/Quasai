#include "quasai/storage/allocator.hpp"
#include <iostream>
#include <new>
#include <stdexcept>

namespace quasai::storage {

size_t CpuAllocator::align(std::size_t size) {
  const size_t alignment = 64; // Cache line size
  return (size + (alignment - 1)) & ~(alignment - 1);
}

void CpuAllocator::split_block(Block *block, std::size_t size) {
  if (block->size < size) {
    throw std::bad_alloc();
  }

  Block *new_block = new Block{static_cast<char *>(block->ptr) + size,
                               block->size - size, false, block, block->next};
  if (new_block->next) {
    new_block->next->prev = new_block;
  } else {
    tail_ = new_block; // Update tail if this is the last block
  }
  block->next = new_block;
  block->size = size;
  block->allocated = true;
}

void *CpuAllocator::allocate(std::size_t size) {
  if (size == 0) {
    size = 1; // Allocate at least 1 byte to ensure unique pointer
  }

  size = align(size);

  // Fast path: check tail first for a large enough free block
  if (!tail_->allocated && tail_->size >= size) {
    Block *b = tail_;
    split_block(b, size);
    allocations_[b->ptr] = b; // Track allocation for deallocation
    return b->ptr;
  }

  for (Block *b = head_; b != nullptr; b = b->next) {
    if (!b->allocated && b->size >= size) {
      split_block(b, size);
      allocations_[b->ptr] = b; // Track allocation for deallocation
      return b->ptr;
    }
  }

  throw std::bad_alloc();
}

void CpuAllocator::deallocate(void *ptr) {
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    throw std::invalid_argument("Invalid pointer deallocation");
  }
  Block *b = it->second;
  b->allocated = false;
  allocations_.erase(ptr); // Remove from tracking

  while (b->next && !b->next->allocated) {
    Block *next = b->next;
    b->size += next->size;
    b->next = next->next;
    if (b->next) {
      b->next->prev = b; // FIX linkage
    }
    delete next;
  }

  while (b->prev && !b->prev->allocated) {
    Block *prev = b->prev;
    prev->size += b->size;
    prev->next = b->next;
    if (b->next) {
      b->next->prev = prev; // FIX linkage
    }
    delete b;
    b = prev; // continue merging from the combined block
  }

  if (!b->next) {
    tail_ = b; // Update tail if we merged to the end
  }
}

CpuAllocator &CpuAllocator::instance() {
  static CpuAllocator allocator;
  return allocator;
}

CpuAllocator::CpuAllocator() {
  // Create a memory pool for allocations
  size_t pool_size = 1024 * 1024 * 1024; // 1 GB
  Block *head = new Block{nullptr, pool_size, false, nullptr, nullptr};

  head->ptr = std::aligned_alloc(64, pool_size);

  if (!head->ptr) {
    throw std::bad_alloc();
  }

  head_ = head;
  tail_ = head;
}

} // namespace quasai::storage
