#pragma once
#include <cmath>
#include "tensor.hh"

namespace upsilon {

class Op {
public:
  std::vector<std::shared_ptr<Op>> inputs;
  Tensor<float> output;
  Tensor<float> grad;

  Op() : output(0), grad(0) {}

  virtual void forward() = 0;
  virtual void backward() = 0;
};


class Variable : public Op {
public:
  Variable(Tensor<float>&& tensor) {
    this->output = tensor;
  }

  void forward() override {
  }

  void backward() override {
  }
};

class Add : public Op {
public:
  Add(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
  inputs.push_back(a);
  inputs.push_back(b);
  }

  void forward() override {
  output = inputs[0]->output.add(inputs[1]->output);
  }

  void backward() override {
  inputs[0]->grad = inputs[0]->grad.add(grad);
  inputs[1]->grad = inputs[1]->grad.add(grad);
  }
};

class Mul : public Op {
public:
  Mul(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
  inputs.push_back(a);
  inputs.push_back(b);
  }

  void forward() override {
  output = inputs[0]->output.mul(inputs[1]->output);
  }

  void backward() override {
  inputs[0]->grad = inputs[0]->grad.add(inputs[1]->output.mul(grad));
  inputs[1]->grad = inputs[1]->grad.add(inputs[0]->output.mul(grad));
  }
};

class Sub : public Op {
public:
  Sub(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
  inputs.push_back(a);
  inputs.push_back(b);
  }

  void forward() override {
  output = inputs[0]->output.sub(inputs[1]->output);
  }

  void backward() override {
  inputs[0]->grad = inputs[0]->grad.add(grad);
  inputs[1]->grad = inputs[1]->grad.sub(grad); // 注意减法的梯度传播
  }
};

class Div : public Op {
public:
  Div(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
  inputs.push_back(a);
  inputs.push_back(b);
  }

  void forward() override {
  output = inputs[0]->output.div(inputs[1]->output);
  }

  void backward() override {
  inputs[0]->grad = inputs[0]->grad.add(grad.mul(inputs[1]->output.inv()));
  inputs[1]->grad = inputs[1]->grad.sub(grad.mul(inputs[0]->output).div(inputs[1]->output.square()));
  }
};

class MatMul : public Op {
public:
  MatMul(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
    inputs.push_back(a);
    inputs.push_back(b);
  }

  void forward() override {
    output = inputs[0]->output.matmul(inputs[1]->output);
  }

  void backward() override {
    // 简化的梯度计算，真实实现需要考虑维度匹配和转置
    inputs[0]->grad = grad.matmul(inputs[1]->output.transposed());
    inputs[1]->grad = inputs[0]->output.transposed().matmul(grad);
  }
};

class Tanh : public Op {
public:
  Tanh(std::shared_ptr<Op> a) {
    inputs.push_back(a);
  }

  void forward() override {
    output = inputs[0]->output.apply([](float x) { return std::tanh(x); });
  }

  void backward() override {
    inputs[0]->grad = inputs[0]->grad.add(grad.mul(output.apply([](float y) { return 1 - y * y; })));
  }
};

class Mul : public Op {
public:
  Mul(std::shared_ptr<Op> a, std::shared_ptr<Op> b) {
    inputs.push_back(a);
    inputs.push_back(b);
  }

  void forward() override {
    output = inputs[0]->output.mul(inputs[1]->output);
  }

  void backward() override {
    inputs[0]->grad = inputs[0]->grad.add(inputs[1]->output.mul(grad));
    inputs[1]->grad = inputs[1]->grad.add(inputs[0]->output.mul(grad));
  }
};

class ReLU : public Op {
public:
  ReLU(std::shared_ptr<Op> a) {
  inputs.push_back(a);
  }

  void forward() override {
  output = inputs[0]->output.apply([](float x) { return std::max(0.0f, x); });
  }

  void backward() override {
  inputs[0]->grad = inputs[0]->grad.add(grad.apply([](float x) { return x > 0 ? 1.0f : 0.0f; }));
  }
};

class Sigmoid : public Op {
public:
  Sigmoid(std::shared_ptr<Op> a) {
  inputs.push_back(a);
  }

  void forward() override {
  output = inputs[0]->output.apply([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
  }

  void backward() override {
  auto sigmoid_grad = output.apply([](float y) { return y * (1 - y); });
  inputs[0]->grad = inputs[0]->grad.add(grad.mul(sigmoid_grad));
  }
};
/*
class Concat : public Op {
public:
  // axis是拼接的维度
  Concat(std::vector<std::shared_ptr<Op>> ops, int axis) {
    this->inputs = std::move(ops);
    this->axis = axis;
  }

  void forward() override {
    output = Tensor<float>::concat(inputs, axis);
  }

  void backward() override {
    // 反向传播时需要将梯度分配回各个输入
    std::vector<upsilon::Tensor<float>> grads = grad.split(inputs.size(), axis);
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputs[i]->grad = inputs[i]->grad.add(grads[i]);
    }
  }

private:
  int axis;
};

class Slice : public Op {
public:
  // start和end是切片的起始和结束索引，这里简化为一维切片
  Slice(std::shared_ptr<Op> a, int start, int end) {
    inputs.push_back(a);
    this->start = start;
    this->end = end;
  }

  void forward() override {
    output = inputs[0]->output.slice(start, end);
  }

  void backward() override {
    // 反向传播时需要将梯度恢复到原始大小，并只在切片部分更新梯度
    // 创建一个与输入相同形状的全零张量
    upsilon::Tensor<float> grad_input = upsilon::Tensor<float>::zeros_like(inputs[0]->output);
    // 将梯度放置在正确的位置
    grad_input.slice(start, end).fill(grad);
    // 更新输入的梯度
    inputs[0]->grad = inputs[0]->grad.add(grad_input);
  }

private:
  int start, end;
};
*/
} // namespace upsilon
