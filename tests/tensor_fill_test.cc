#include "tensor.hh"

#include <gtest/gtest.h>

// TEST(TensorFillTest, fill1) {
//   upsilon::Tensor<float> f1(upsilon::TensorType::Tensor, {2, 3, 4});
//   std::vector<float> values(2 * 3 * 4);
//   for (int i = 0; i < 24; ++i) {
//     values.at(i) = float(i + 1);
//   }
//   std::cout << "Values:\n";
//   for (int i = 0; i < 24; ++i) {
//     std::cout << values.at(i) << " ";
//   }
//   f1.fill(values);
//   f1.show();
//   int i = 0;
//   for (int c = 0; c < 2; ++c) {
//     for (int x = 0; x < 3; ++x) {
//       for (int y = 0; y < 4; ++y) {
//         EXPECT_EQ(f1.at(c, x, y), values.at(i));
//         i++;
//       }
//     }
//   }
// }

TEST(TensorFillTest, reshape1) {
  upsilon::Tensor<float> f1(upsilon::TensorType::Tensor, {2, 3, 4});
  std::vector<float> values(2 * 3 * 4);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }
  f1.fill(values);
  f1.show();

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(f1.at(i, j, k), values.at(i * 3 * 4 + j * 4 + k));
      }
    }
  }

  f1.reshape({4, 3, 2});
  f1.show();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(f1.at(i, j, k), values.at(i * 3 * 2 + j * 2 + k));
      }
    }
  }
}
