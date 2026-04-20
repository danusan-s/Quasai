#include <cstddef>
#include <cstdlib>

namespace quasai {

class Allocator {
public:
  virtual void *allocate(std::size_t size) = 0;
  virtual void deallocate(void *ptr) = 0;

  Allocator(const Allocator &) = delete;
  Allocator &operator=(const Allocator &) = delete;
  ~Allocator() = default;

protected:
  Allocator() = default;
};

class CpuAllocator : public Allocator {
public:
  void *allocate(std::size_t size) override {
    if (size == 0) {
      size = 1; // malloc(0) behavior is implementation-defined, so we allocate
                // at least 1 byte
    }
    return std::malloc(size);
  }
  void deallocate(void *ptr) override {
    std::free(ptr);
  }

  static CpuAllocator &instance() {
    static CpuAllocator allocator;
    return allocator;
  }
};

class GpuAllocator : public Allocator {
  // TODO: Implement GPU memory allocation and deallocation
};

} // namespace quasai
