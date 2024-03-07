#pragma once
#include "tensor.hh"

namespace upsilon {

  class Fn {
  public:
    virtual Tensor<float> forward(const Tensor<float> &x, const Tensor<float> &y) = 0;
    virtual std::tuple<Tensor<float>,Tensor<float>> backward(const Tensor<float> &grad) = 0;
    virtual std::string name() = 0;
  };

  class AddFn: public Fn {
  public:
    Tensor<float> forward(const Tensor<float> &x, const Tensor<float> &y) {
      Tensor<float> result(x.shape());
      for (uint32_t i = 0; i < x.size(); ++i) {
        auto sum = x.at(i) + y.at(i);
        result.at(i) = sum;
      }
      return result;
    }

    std::tuple<Tensor<float>,Tensor<float>> backward(const Tensor<float> &grad) {
      return std::make_tuple(grad, grad);
    }

    std::string name() {
      return "AddFn";
    }
  };

  class MulFn: public Fn {

    std::tuple<Tensor<float>, Tensor<float>> saved_tensors_ = std::make_tuple(Tensor<float>(), Tensor<float>());
  public:
    Tensor<float> forward(const Tensor<float> &x, const Tensor<float> &y) {
      Tensor<float> result(x.shape());
      for (uint32_t i = 0; i < x.size(); ++i) {
        auto product = x.at(i) * y.at(i);
        result.at(i) = product;
      }
      saved_tensors_ = std::make_tuple(x, y);
      return result;
    }

    // 这里的 grad 是损失函数对输出 z 的梯度，
    // 即 ∂L/∂z，乘以相应的导数 y 和 x，即 ∂z/∂x 和 ∂z/∂y，来计算 ∂L/∂x 和 ∂L/∂y。
    std::tuple<Tensor<float>,Tensor<float>> backward(const Tensor<float> &grad) {
      auto x = std::get<0>(saved_tensors_);
      auto y = std::get<1>(saved_tensors_);
      Tensor<float> dx = grad.mul(y);
      Tensor<float> dy = grad.mul(x);
      return std::make_tuple(dx, dy);
    }

    std::string name() {
      return "MulFn";
    }
  };
}

