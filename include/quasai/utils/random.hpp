#pragma once

#include <random>

namespace quasai::utils {

/**
 * @brief Singleton random number generator (Mersenne Twister).
 * @note Default seed is 42 for reproducibility.
 */
class RNG {
public:
  /// @brief Get the singleton instance.
  static RNG &instance() {
    static RNG rng;
    return rng;
  }

  /// @brief Get the underlying mt19937 engine.
  std::mt19937 &engine() {
    return engine_;
  }

  /// @brief Seed the random number generator.
  void seed(uint32_t seed) {
    engine_.seed(seed);
  }

private:
  /// @brief Private constructor with default seed.
  RNG() : engine_(42) { // Default seed for reproducibility
  }
  std::mt19937 engine_; ///< Mersenne Twister engine
};

} // namespace quasai::utils
