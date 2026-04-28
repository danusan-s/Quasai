#include "quasai/storage/allocator.hpp"
#include <gtest/gtest.h>

using namespace quasai::storage;

TEST(MemoryTest, CpuAllocator) {
  CpuAllocator allocator;

  // Test allocation and deallocation
  void *ptr = allocator.allocate(1024);
  EXPECT_NE(ptr, nullptr);                    // Allocation should succeed
  EXPECT_NO_THROW(allocator.deallocate(ptr)); // Should not throw

  // Test zero-size allocation
  void *zero_ptr = allocator.allocate(0);
  EXPECT_NE(zero_ptr, nullptr); // Should still allocate at least 1 byte
  EXPECT_NO_THROW(allocator.deallocate(zero_ptr)); // Should not throw
}
