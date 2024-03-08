#include "tensor.hh"

#include <gtest/gtest.h>

TEST(TensorGetSizeTest, size1) {
  upsilon::Tensor<float> f1(upsilon::TensorType::Tensor, {2, 3, 4});
  EXPECT_EQ(f1.size(), 24);
  EXPECT_EQ(f1.channels(), 2);
  EXPECT_EQ(f1.rows(), 3);
  EXPECT_EQ(f1.cols(), 4);
}

TEST(TensorGetValueTest, GetValue) {
  upsilon::Tensor<float> f1(upsilon::TensorType::Tensor, {2, 3, 4});
  f1.fill(1.f);
  EXPECT_EQ(f1.at(1, 2, 3), 1.f);

  upsilon::Tensor<float> f2(upsilon::TensorType::Tensor, {4, 3, 2});
  std::vector<float> values(4 * 3 * 2);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }

  f2.fill(values);
  EXPECT_EQ(f2.at(1, 2, 1), values.at(1 * 3 * 2 + 2 * 2 + 1));
}
