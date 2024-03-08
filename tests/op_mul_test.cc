#include <gtest/gtest.h>
#include "tensor.hh"
#include "op.hh"

TEST(MulFnTest, BasicMultiplication) {
    using namespace upsilon;
    
    Tensor<float> x({2, 2});
    Tensor<float> y({2, 2});

    x.fill({1, 2, 3, 4});
    y.fill({5, 6, 7, 8});


    MulFn mul_fn;
    Tensor<float> result = mul_fn.forward(x, y);

    ASSERT_FLOAT_EQ(result.at(0), 5); // 1 * 5 = 5
    ASSERT_FLOAT_EQ(result.at(1), 12); // 2 * 6 = 12
    ASSERT_FLOAT_EQ(result.at(2), 21); // 3 * 7 = 21
    ASSERT_FLOAT_EQ(result.at(3), 32); // 4 * 8 = 32
}

TEST(MulFnTest, GradientCalculation) {
    using namespace upsilon;

    Tensor<float> x({2, 2});
    Tensor<float> y({2, 2});

    x.fill({1, 2, 3, 4});
    y.fill({5, 6, 7, 8});

    Tensor<float> grad({2, 2}); // Loss gradient with respect to output
    grad.fill({2, 3, 4, 5});

    // Instantiate MulFn
    MulFn mul_fn;

    // Forward pass
    Tensor<float> result = mul_fn.forward(x, y);

    // Backward pass
    auto gradients = mul_fn.backward(grad);
    Tensor<float> dx = std::get<0>(gradients); // Gradient with respect to x
    Tensor<float> dy = std::get<1>(gradients); // Gradient with respect to y

    // Expected gradients
    std::vector<float> expected_dx({10, 18, 28, 40}); // dx = grad * y
    std::vector<float> expected_dy({2, 6, 12, 20});  // dy = grad * x

    // Check gradients
    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_FLOAT_EQ(dx.at(i), expected_dx.at(i));
        ASSERT_FLOAT_EQ(dy.at(i), expected_dy.at(i));
    }
}
