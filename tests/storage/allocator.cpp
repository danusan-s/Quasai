#include "quasai/storage/allocator.hpp"
#include <cstring>
#include <gtest/gtest.h>

using namespace quasai::storage;

TEST(CpuAllocatorTest, SingletonInstance) {
  CpuAllocator &a1 = CpuAllocator::instance();
  CpuAllocator &a2 = CpuAllocator::instance();
  EXPECT_EQ(&a1, &a2); // Same singleton instance
}

TEST(CpuAllocatorTest, BasicAllocation) {
  CpuAllocator &allocator = CpuAllocator::instance();

  void *ptr = allocator.allocate(1024);
  EXPECT_NE(ptr, nullptr);
  allocator.deallocate(ptr);
}

TEST(CpuAllocatorTest, ZeroSizeAllocation) {
  CpuAllocator &allocator = CpuAllocator::instance();

  void *ptr = allocator.allocate(0);
  EXPECT_NE(ptr, nullptr); // Should allocate at least 1 byte after alignment
  allocator.deallocate(ptr);
}

TEST(CpuAllocatorTest, MultipleAllocations) {
  CpuAllocator &allocator = CpuAllocator::instance();

  void *p1 = allocator.allocate(100);
  void *p2 = allocator.allocate(200);
  void *p3 = allocator.allocate(300);

  EXPECT_NE(p1, nullptr);
  EXPECT_NE(p2, nullptr);
  EXPECT_NE(p3, nullptr);
  EXPECT_NE(p1, p2);
  EXPECT_NE(p2, p3);

  allocator.deallocate(p1);
  allocator.deallocate(p2);
  allocator.deallocate(p3);
}

TEST(CpuAllocatorTest, Alignment) {
  CpuAllocator &allocator = CpuAllocator::instance();

  // Allocate small sizes and check alignment to 64 bytes
  for (size_t size : {1, 31, 63, 64, 65, 127, 128}) {
    void *ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0)
        << "Pointer not aligned to 64 bytes for size " << size;
    allocator.deallocate(ptr);
  }
}

TEST(CpuAllocatorTest, WriteAndRead) {
  CpuAllocator &allocator = CpuAllocator::instance();

  size_t size = 1024;
  void *ptr = allocator.allocate(size);
  EXPECT_NE(ptr, nullptr);
  // Write pattern and verify
  memset(ptr, 0xAB, size);
  unsigned char *data = static_cast<unsigned char *>(ptr);
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(data[i], 0xAB);
  }

  allocator.deallocate(ptr);
}

TEST(CpuAllocatorTest, Coalescing) {
  CpuAllocator &allocator = CpuAllocator::instance();

  // Allocate three blocks
  void *p1 = allocator.allocate(100);
  void *p2 = allocator.allocate(200);
  void *p3 = allocator.allocate(300);

  // Free middle block
  allocator.deallocate(p2);

  // Now free p1 and p3 - they should coalesce with p2 if adjacent
  allocator.deallocate(p1);
  allocator.deallocate(p3);

  // After all deallocations, we should be able to allocate a large block
  void *big = allocator.allocate(500);
  EXPECT_NE(big, nullptr);
  allocator.deallocate(big);
}

TEST(CpuAllocatorTest, AllocateDeallocateCycle) {
  CpuAllocator &allocator = CpuAllocator::instance();

  // Multiple cycles of allocate/deallocate
  for (int cycle = 0; cycle < 10; ++cycle) {
    void *ptrs[5];
    for (int i = 0; i < 5; ++i) {
      ptrs[i] = allocator.allocate(100 * (i + 1));
      EXPECT_NE(ptrs[i], nullptr);
    }
    for (int i = 4; i >= 0; --i) {
      allocator.deallocate(ptrs[i]);
    }
  }
}

TEST(CpuAllocatorTest, InvalidDeallocation) {
  CpuAllocator &allocator = CpuAllocator::instance();

  // Try to deallocate an invalid pointer
  void *invalid = reinterpret_cast<void *>(0x12345678);
  EXPECT_THROW(allocator.deallocate(invalid), std::invalid_argument);
}

TEST(CpuAllocatorTest, LargeAllocation) {
  CpuAllocator &allocator = CpuAllocator::instance();

  // Try to allocate something large (but within the 1GB pool)
  void *ptr = allocator.allocate(10 * 1024 * 1024); // 10 MB
  EXPECT_NE(ptr, nullptr);
  allocator.deallocate(ptr);
}
