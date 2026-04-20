#include "quasai/core/memory.hpp"
#include <gtest/gtest.h>

TEST(MemoryTest, CpuAllocator) {
  quasai::CpuAllocator allocator;

  // Test allocation and deallocation
  void *ptr = allocator.allocate(1024);
  EXPECT_NE(ptr, nullptr);                    // Allocation should succeed
  EXPECT_NO_THROW(allocator.deallocate(ptr)); // Should not throw

  // Test zero-size allocation
  void *zero_ptr = allocator.allocate(0);
  EXPECT_NE(zero_ptr, nullptr); // Should still allocate at least 1 byte
  EXPECT_NO_THROW(allocator.deallocate(zero_ptr)); // Should not throw
}

TEST(BufferTest, CpuBuffer) {
  quasai::CpuAllocator allocator;

  // Test buffer creation and destruction
  {
    quasai::Buffer buffer(&allocator, 1024);

    EXPECT_NE(buffer.raw_data(),
              nullptr);             // Buffer should have allocated memory
    EXPECT_EQ(buffer.size(), 1024); // Buffer size should be correct
  } // Buffer goes out of scope here, should deallocate memory

  // Test zero-size buffer
  {
    quasai::Buffer zero_buffer(&allocator, 0);
    EXPECT_NE(zero_buffer.raw_data(),
              nullptr);               // Should still allocate at least 1 byte
    EXPECT_EQ(zero_buffer.size(), 0); // Size should be reported as 0
  } // Zero buffer goes out of scope here, should deallocate memory
}
