#include <Eigen/Dense>

// 定义tensor，这里以Eigen的Matrix作为示例
using tensor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class Add {
 public:
  tensor forward(tensor x, tensor y) {
    return x + y;  // 前向传播，直接相加
  }

  std::pair<tensor, tensor> backward(tensor dout) {
    // 反向传播，因为加法是对称的，所以两个输入的梯度相同
    return {dout, dout};  // 返回关于x和y的梯度
  }
};
