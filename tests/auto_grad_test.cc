#include <gtest/gtest.h>
#include "op.hh" // Replace with the actual file name

using namespace upsilon;

TEST(AutomaticDifferentiationTest, GradientCheck) {
    // 创建变量
    auto a = std::make_shared<Variable>(Tensor<float>({1}, 3.0f));
    auto b = std::make_shared<Variable>(Tensor<float>({1}, 2.0f));

    // 构建计算图
    auto mul_ab = std::make_shared<Mul>(a, b);
    auto add_a_ab = std::make_shared<Add>(a, mul_ab);

    // 正向传播
    mul_ab->forward();
    add_a_ab->forward();

    // 反向传播
    add_a_ab->grad = Tensor<float>({1}, 1.0f); // 对 y 的梯度是 1
    add_a_ab->backward();
    mul_ab->backward();

    // 输出结果和梯度
    std::cout << "Result: " << add_a_ab->output.data() << std::endl;
    std::cout << "Grad a: " << a->grad.data() << std::endl;
    std::cout << "Grad b: " << b->grad.data() << std::endl;

}
