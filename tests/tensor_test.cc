#include "tensor.hh"

#include <gtest/gtest.h>

TEST(TensorTest, CreateTensor) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  EXPECT_EQ(tensor.channels(), 3);
  EXPECT_EQ(tensor.rows(), 4);
  EXPECT_EQ(tensor.cols(), 5);
}

TEST(TensorTest, CreateTensorWithVector) {
  auto tensor = upsilon::Tensor<float>::create({3, 4, 5});
  EXPECT_EQ(tensor.channels(), 3);
  EXPECT_EQ(tensor.rows(), 4);
  EXPECT_EQ(tensor.cols(), 5);
}

TEST(TensorTest, CreateTensorWithOneDimension) {
  auto tensor = upsilon::Tensor<float>::create({3});
  EXPECT_EQ(tensor.channels(), 1);
  EXPECT_EQ(tensor.rows(), 1);
  EXPECT_EQ(tensor.cols(), 3);
}

TEST(TensorTest, CreateTensorWithTwoDimensions) {
  auto tensor = upsilon::Tensor<float>::create({3, 4});
  EXPECT_EQ(tensor.channels(), 1);
  EXPECT_EQ(tensor.rows(), 3);
  EXPECT_EQ(tensor.cols(), 4);
}

TEST(TensorTest, CreateTensorWithInvalidShape) {
  EXPECT_THROW(upsilon::Tensor<float>::create({3, 4, 5, 6}),
               std::invalid_argument);
}
