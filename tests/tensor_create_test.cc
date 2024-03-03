#include "tensor.hh"

#include <gtest/gtest.h>

TEST(TensorTest, tensor_init1D) {
  upsilon::Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &shape = f1.shape();
  const uint32_t size = shape.at(0);
  f1.Show();

  EXPECT_EQ(size, 4);

}

TEST(TensorTest, tensor_init2D) {
  upsilon::Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &shape = f1.shape();
  const uint32_t rows = shape.at(0);
  const uint32_t cols = shape.at(1);

  f1.Show();
  EXPECT_EQ(rows, 4);
  EXPECT_EQ(cols, 4);

}

TEST(TensorTest, tensor_init3D_3) {
  upsilon::Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &shape = f1.shape();
  const uint32_t channels = shape.at(0);
  const uint32_t rows = shape.at(1);
  const uint32_t cols = shape.at(2);

  f1.Show();
  EXPECT_EQ(channels, 2);
  EXPECT_EQ(rows, 3);
  EXPECT_EQ(cols, 4);
}

TEST(TensorTest, tensor_init3D_2) {
  upsilon::Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &shape = f1.shape();
  const uint32_t rows = shape.at(0);
  const uint32_t cols = shape.at(1);

  EXPECT_EQ(rows, 1);
  EXPECT_EQ(cols, 2);

  f1.Show();
}

TEST(TensorTest, tensor_init3D_1) {
  upsilon::Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &shape = f1.shape();
  const uint32_t size = shape.at(0);

  EXPECT_EQ(size, 1);

  f1.Show();
}
