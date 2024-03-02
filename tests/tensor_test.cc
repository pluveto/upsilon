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

TEST(TensorTest, SetData) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  std::vector<Eigen::MatrixXf> data(3, Eigen::MatrixXf(4, 5));
  tensor.set_data(data);
}

TEST(TensorTest, Size) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  EXPECT_EQ(tensor.size(), 60);
}

TEST(TensorTest, Index) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  std::vector<Eigen::MatrixXf> data(3, Eigen::MatrixXf(4, 5));
  tensor.set_data(data);
  EXPECT_EQ(tensor.index(0), 0);
  tensor.index(0) = 1;
  EXPECT_EQ(tensor.index(0), 1);

  EXPECT_EQ(tensor.index(4), 0);
  tensor.index(4) = 1;
  EXPECT_EQ(tensor.index(4), 1);
}

TEST(TensorTest, Fill) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  tensor.fill(1);
  for (uint32_t i = 0; i < tensor.size(); i++) {
    EXPECT_EQ(tensor.index(i), 1);
  }
}

TEST(TensorTest, TestEnsureRowMajor) {
  upsilon::Tensor<float> tensor(3, 4, 5);
  // fill 1~60
  for (uint32_t i = 0; i < tensor.size(); i++) {
    tensor.index(i) = i + 1;
  }
  // access the data in row-major order
  for (uint32_t i = 0; i < tensor.rows(); i++) {
    for (uint32_t j = 0; j < tensor.cols(); j++) {
      for (uint32_t k = 0; k < tensor.channels(); k++) {
        EXPECT_EQ(tensor.at(k, i, j), i * tensor.cols() + j + 1 + k * 20);
      }
    }
  }
}
