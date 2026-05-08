#pragma once

namespace quasai::core {

/// @brief Supported device types.
enum class DeviceType {
  CPU,      ///< CPU device
  GPU_CUDA, ///< CUDA GPU device
};

/**
 * @brief Represents a device where a tensor is allocated.
 */
struct Device {
  DeviceType type_; ///< Device type (CPU or GPU_CUDA)
  int id_; ///< For GPU devices, this is the GPU index. For CPU, this is
           ///< ignored.

  /// @brief Factory method for creating a CPU device.
  static Device cpu() {
    return Device{DeviceType::CPU, 0};
  }

  /// @brief Factory method for creating a GPU device.
  static Device gpu(int gpu_id) {
    return Device{DeviceType::GPU_CUDA, gpu_id};
  }
};

} // namespace quasai::core
