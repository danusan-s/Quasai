namespace quasai {

typedef enum {
  CPU,
  GPU_CUDA,
} DeviceType;

struct Device {
  DeviceType type;
  int id; // For GPU devices, this is the GPU index. For CPU, this can be
          // ignored.

  // Factory method for creating a CPU device
  Device cpu() {
    return Device{CPU, 0};
  }
};

} // namespace quasai
