#pragma once
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

  class Add: public Op {
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

  class Mul: public Op {
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
}
