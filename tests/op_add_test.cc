#include <gtest/gtest.h>
#include "tensor.hh"
#include "op.hh"


TEST(AddFnTest, BasicAddition) {
    using namespace upsilon;

    Tensor<float> x({2, 2});
    Tensor<float> y({2, 2});

    // fill 1~8
    for (int i = 0; i < 4; ++i) {
        x.at(i) = i + 1;
        y.at(i) = i + 5;
    }

    AddFn add_fn;
    Tensor<float> result = add_fn.forward(x, y);

    ASSERT_FLOAT_EQ(result.at(0), 6); // 1 + 5 = 6
    ASSERT_FLOAT_EQ(result.at(1), 8); // 2 + 6 = 8
    ASSERT_FLOAT_EQ(result.at(2), 10); // 3 + 7 = 10
    ASSERT_FLOAT_EQ(result.at(3), 12); // 4 + 8 = 12
}

TEST(AddFnTest, BasicAdditionBackward) {
    using namespace upsilon;

    Tensor<float> grad({2, 2});
    
    // fill 1~4
    for (int i = 0; i < 4; ++i) {
        grad.at(i) = i + 1;
    }

    AddFn add_fn;
    auto result = add_fn.backward(grad);
    auto x_grad = std::get<0>(result);
    auto y_grad = std::get<1>(result);

    ASSERT_FLOAT_EQ(x_grad.at(0), 1); // 1 * 1 = 1
    ASSERT_FLOAT_EQ(x_grad.at(1), 2); // 2 * 1 = 2
    ASSERT_FLOAT_EQ(x_grad.at(2), 3); // 3 * 1 = 3
    ASSERT_FLOAT_EQ(x_grad.at(3), 4); // 4 * 1 = 4

    ASSERT_FLOAT_EQ(y_grad.at(0), 1); // 1 * 1 = 1
    ASSERT_FLOAT_EQ(y_grad.at(1), 2); // 2 * 1 = 2
    ASSERT_FLOAT_EQ(y_grad.at(2), 3); // 3 * 1 = 3
    ASSERT_FLOAT_EQ(y_grad.at(3), 4); // 4 * 1 = 4
}
