#include "quasai/nn/module.hpp"

namespace quasai {

class Sequential : public Module {
public:
  Sequential(const std::vector<std::shared_ptr<Module>> &modules);

  Tensor forward(const Tensor &input) override;

private:
  std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace quasai
