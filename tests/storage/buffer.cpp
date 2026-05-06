#include "quasai/storage/buffer.hpp"
#include <cstring>
#include <gtest/gtest.h>

using namespace quasai::storage;

TEST(BufferTest, CreateAndDestroy) {
  CpuAllocator &allocator = CpuAllocator::instance();

  {
    Buffer buffer(&allocator, 1024);
    EXPECT_NE(buffer.raw_data(), nullptr);
    EXPECT_EQ(buffer.size(), 1024);
  } // Buffer should deallocate on destruction
}

TEST(BufferTest, ZeroSizeBuffer) {
  CpuAllocator &allocator = CpuAllocator::instance();

  Buffer buffer(&allocator, 0);
  EXPECT_NE(buffer.raw_data(), nullptr); // Aligned allocation still returns ptr
  EXPECT_EQ(buffer.size(), 0);
}

TEST(BufferTest, DefaultAllocator) {
  // Test with explicit default
  Buffer buffer2(&CpuAllocator::instance(), 512);
  EXPECT_NE(buffer2.raw_data(), nullptr);
  EXPECT_EQ(buffer2.size(), 512);
}

TEST(BufferTest, WriteAndRead) {
  CpuAllocator &allocator = CpuAllocator::instance();
  Buffer buffer(&allocator, 1024);

  void *data = buffer.raw_data();
  memset(data, 0xAB, buffer.size());

  unsigned char *bytes = static_cast<unsigned char *>(data);
  for (size_t i = 0; i < buffer.size(); ++i) {
    EXPECT_EQ(bytes[i], 0xAB);
  }
}

TEST(BufferTest, MoveConstructor) {
  CpuAllocator &allocator = CpuAllocator::instance();

  Buffer buffer1(&allocator, 1024);
  void *original_ptr = buffer1.raw_data();
  EXPECT_NE(original_ptr, nullptr);

  Buffer buffer2(std::move(buffer1));
  EXPECT_EQ(buffer2.raw_data(), original_ptr); // Same memory
  EXPECT_EQ(buffer2.size(), 1024);

  // buffer1 should be moved from (in a valid but unspecified state)
  // Typically raw_data() might be nullptr after move
}

TEST(BufferTest, MoveAssignment) {
  CpuAllocator &allocator = CpuAllocator::instance();

  Buffer buffer1(&allocator, 512);
  void *original_ptr = buffer1.raw_data();

  Buffer buffer2(&allocator, 256);
  buffer2 = std::move(buffer1);

  EXPECT_EQ(buffer2.raw_data(), original_ptr);
  EXPECT_EQ(buffer2.size(), 512);
}

TEST(BufferTest, MultipleBuffers) {
  CpuAllocator &allocator = CpuAllocator::instance();

  Buffer buffers[5];
  for (int i = 0; i < 5; ++i) {
    buffers[i] = Buffer(&allocator, (i + 1) * 100);
    EXPECT_NE(buffers[i].raw_data(), nullptr);
    EXPECT_EQ(buffers[i].size(), (i + 1) * 100);
  }

  // All pointers should be different
  for (int i = 0; i < 5; ++i) {
    for (int j = i + 1; j < 5; ++j) {
      EXPECT_NE(buffers[i].raw_data(), buffers[j].raw_data());
    }
  }
}

TEST(BufferTest, BufferScope) {
  CpuAllocator &allocator = CpuAllocator::instance();
  void *ptr = nullptr;

  {
    Buffer buffer(&allocator, 2048);
    ptr = buffer.raw_data();
    EXPECT_NE(ptr, nullptr);
  } // buffer destroyed here, memory deallocated

  // After deallocation, we can allocate again
  Buffer buffer2(&allocator, 2048);
  EXPECT_NE(buffer2.raw_data(), nullptr);
  // ptr might be reused, which is fine
}

TEST(BufferTest, SizeConsistency) {
  CpuAllocator &allocator = CpuAllocator::instance();

  for (size_t size : {0, 1, 63, 64, 65, 1000, 10000}) {
    Buffer buffer(&allocator, size);
    EXPECT_EQ(buffer.size(), size);
  }
}
