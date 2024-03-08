#include <gtest/gtest.h>
#include "op.hh" // Replace with the actual file name

using namespace upsilon;

TEST(AutomaticDifferentiationTest, GradientCheck) {
    auto a = std::make_shared<Variable>(Tensor<float>(3.0f));
    auto b = std::make_shared<Variable>(Tensor<float>(2.0f));
    std::cout << "a: " << a->output << std::endl;
    std::cout << "b: " << b->output << std::endl;

    auto mul_ab = std::make_shared<Mul>(a, b);
    auto add_a_ab = std::make_shared<Add>(a, mul_ab);

    std::cout << "start forward" << std::endl;
    mul_ab->forward();
    add_a_ab->forward();

    std::cout << "start backward" << std::endl;
    add_a_ab->grad = Tensor<float>(1.0f); // 对 y 的梯度是 1
    add_a_ab->backward();
    mul_ab->backward();

    std::cout << "Result: " << add_a_ab->output << std::endl;
    std::cout << "Grad a: " << a->grad << std::endl;
    std::cout << "Grad b: " << b->grad << std::endl;

}
