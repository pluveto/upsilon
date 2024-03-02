#include "tensor.hh"

using tensor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class Add {
 public:
  tensor forward(tensor x, tensor y) { return x + y; }

  std::pair<tensor, tensor> backward(tensor dout) { return {dout, dout}; }
};
