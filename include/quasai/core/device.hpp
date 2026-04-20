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
  static Device cpu() {
    return Device{CPU, 0};
  }

  static Device gpu(int gpu_id) {
    return Device{GPU_CUDA, gpu_id};
  }
};

} // namespace quasai
