#include "tensor.hh"

#include <gtest/gtest.h>

TEST(TensorTransformTest, neg1) {
  upsilon::Tensor<float> f1(2, 3, 4);
  f1.Fill(2.f);
  
  const auto neg1 = [](float x) { return -1 * x; };
  f1.Transform(neg1);

  for (int c = 0; c < 2; ++c) {
    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 4; ++y) {
        EXPECT_EQ(f1.at(c, x, y), -2.f);
      }
    }
  }
}

TEST(TensorTransformTest, TransposeSquareMatrix) {
  upsilon::Tensor<float> f1(2, 2);
  EXPECT_EQ(f1.channels(), 1);
  EXPECT_EQ(f1.shape().size(), 2);

  std::vector<float> values(2 * 2);
  for (int i = 0; i < 4; ++i) {
    values.at(i) = float(i + 1);
  }

  f1.Fill(values);
  f1.Show();

  f1.Transpose();
  f1.Show();

  EXPECT_EQ(f1.at(0, 0), 1.f);
  EXPECT_EQ(f1.at(0, 1), 3.f);
  EXPECT_EQ(f1.at(1, 0), 2.f);
  EXPECT_EQ(f1.at(1, 1), 4.f);
}

// TEST(TensorTransformTest, TransposeMatrix) {
//   upsilon::Tensor<float> f1(2, 3);
//   std::vector<float> values(2 * 3);
//   for (int i = 0; i < 6; ++i) {
//     values.at(i) = float(i + 1);
//   }

//   f1.Fill(values);
//   f1.Show();

//   f1.Transpose();
//   f1.Show();

//   EXPECT_EQ(f1.at(0, 0), 1.f);
//   EXPECT_EQ(f1.at(0, 1), 4.f);
//   EXPECT_EQ(f1.at(0, 2), 2.f);
//   EXPECT_EQ(f1.at(1, 0), 3.f);
//   EXPECT_EQ(f1.at(1, 1), 6.f);
//   EXPECT_EQ(f1.at(1, 2), 5.f);
// }
