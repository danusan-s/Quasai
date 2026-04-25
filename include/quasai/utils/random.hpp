#pragma once

#include <random>

namespace quasai {

class RNG {
public:
  static RNG &instance() {
    static RNG rng;
    return rng;
  }

  std::mt19937 &engine() {
    return engine_;
  }

  void seed(uint32_t seed) {
    engine_.seed(seed);
  }

private:
  RNG() : engine_(std::random_device{}()) {
    seed(42); // Default seed for reproducibility
  }
  std::mt19937 engine_;
};

} // namespace quasai
