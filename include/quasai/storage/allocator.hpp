#pragma once

#include <cstddef>
#include <cstdlib>

namespace quasai {

class Allocator {
public:
  virtual void *allocate(std::size_t size) = 0;
  virtual void deallocate(void *ptr) = 0;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;
  virtual ~Allocator() = default;

protected:
  Allocator() = default;
};

class CpuAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override;
  void deallocate(void *ptr) override;

  static CpuAllocator &instance();
};

class CudaAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override;
  void deallocate(void *ptr) override;

  static CudaAllocator &instance();
};

} // namespace quasai
