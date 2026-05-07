#include "quasai/storage/allocator.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <new>
#include <stdexcept>

namespace quasai::storage {

static constexpr size_t POOL_SIZE = 1024 * 1024 * 1024; // 1 GB
static constexpr size_t ALIGNMENT = 64;                 // Cache line size

static size_t align(std::size_t size) {
  const size_t alignment = ALIGNMENT;
  return (size + (alignment - 1)) & ~(alignment - 1);
}

static void add_to_bin(std::array<std::vector<Block *>, NUM_BINS> &bins,
                       Block *block) {
  assert(block->allocated == false);
  size_t ind = block->size / ALIGNMENT;
  if (ind < NUM_BINS) {
    bins[ind].push_back(block);
  }
}

void CpuAllocator::split_allocate(Block *block, std::size_t size) {
  if (block->size == size) {
    block->allocated = true;
    allocations_[block->ptr] = block; // Track allocation for deallocation
    return;
  }

  Block *new_block = new Block{static_cast<char *>(block->ptr) + size,
                               block->size - size, false, block, block->next};
  if (new_block->next) {
    new_block->next->prev = new_block;
  } else {
    tail_ = new_block; // Update tail if this is the last block
  }
  add_to_bin(free_bins_, new_block); // Add the new free block to bins
  block->next = new_block;
  block->size = size;
  block->allocated = true;
  allocations_[block->ptr] = block; // Track allocation for deallocation
}

static void remove_from_bin(std::array<std::vector<Block *>, NUM_BINS> &bins,
                            Block *block) {
  assert(block->allocated == false);
  size_t ind = block->size / ALIGNMENT;
  if (ind < NUM_BINS) {
    auto &bin = bins[ind];
    auto it = std::find(bin.begin(), bin.end(), block);
    if (it != bin.end()) {
      *it = bin.back();
      bin.pop_back();
    }
  }
}

void *CpuAllocator::allocate(std::size_t size) {
  if (size == 0) {
    size = 1; // Allocate at least 1 byte to ensure unique pointer
  }

  size = align(size);
  size_t bin_ind = size / ALIGNMENT;

  // Fast path: check free bins for a suitable block
  // If really large, take last big block. Dont put tail in bins
  if (bin_ind < NUM_BINS) {
    for (size_t ind = bin_ind; ind < NUM_BINS; ++ind) {
      if (!free_bins_[ind].empty()) {
        Block *b = free_bins_[ind].back();
        free_bins_[ind].pop_back();
        split_allocate(b, size);
        return b->ptr;
      }
    }
  }

  // Fast path: check tail block for a large enough free block
  if (!tail_->allocated && tail_->size >= size) {
    Block *b = tail_;
    split_allocate(b, size);
    return b->ptr;
  }

  // Slow path: scan from head and coalesce on the fly
  for (Block *b = head_; b; b = b->next) {
    if (!b->allocated) {
      remove_from_bin(free_bins_, b);

      while (b->next && !b->next->allocated) {
        Block *next = b->next;
        remove_from_bin(free_bins_,
                        next); // Remove free block from bins before merging
        b->size += next->size;
        b->next = next->next;
        if (b->next) {
          b->next->prev = b; // FIX linkage
        }
        delete next;
      }

      if (!b->next) {
        tail_ = b; // Update tail if we merged to the end
      }

      if (b->size >= size) {
        split_allocate(b, size);
        return b->ptr;
      }
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

  add_to_bin(free_bins_, b);
}

CpuAllocator &CpuAllocator::instance() {
  static CpuAllocator allocator;
  return allocator;
}

CpuAllocator::CpuAllocator() {
  // Create a memory pool for allocations
  size_t pool_size = POOL_SIZE;
  Block *head = new Block{nullptr, pool_size, false, nullptr, nullptr};

  head->ptr = std::aligned_alloc(ALIGNMENT, pool_size);

  if (!head->ptr) {
    throw std::bad_alloc();
  }

  head_ = head;
  tail_ = head;
}

CpuAllocator::~CpuAllocator() {
  // Free the entire memory pool
  free(head_->ptr);
  Block *current = head_;
  while (current) {
    Block *next = current->next;
    delete current;
    current = next;
  }
}

} // namespace quasai::storage
