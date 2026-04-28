#include "quasai/storage/buffer.hpp"
#include <gtest/gtest.h>

using namespace quasai::storage;

TEST(BufferTest, CpuBuffer) {
  CpuAllocator allocator;

  // Test buffer creation and destruction
  {
    Buffer buffer(&allocator, 1024);

    EXPECT_NE(buffer.raw_data(),
              nullptr);             // Buffer should have allocated memory
    EXPECT_EQ(buffer.size(), 1024); // Buffer size should be correct
  } // Buffer goes out of scope here, should deallocate memory

  // Test zero-size buffer
  {
    Buffer zero_buffer(&allocator, 0);
    EXPECT_NE(zero_buffer.raw_data(),
              nullptr);               // Should still allocate at least 1 byte
    EXPECT_EQ(zero_buffer.size(), 0); // Size should be reported as 0
  } // Zero buffer goes out of scope here, should deallocate memory
}
