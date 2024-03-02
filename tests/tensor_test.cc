#include "tensor.hh"

#include <gtest/gtest.h>

TEST(TensorTest, CreateTensor) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  EXPECT_EQ(tensor.channels(), 3);
  EXPECT_EQ(tensor.rows(), 4);
  EXPECT_EQ(tensor.cols(), 5);
}
