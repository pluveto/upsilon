#include "src/core/model.h"
#include <iostream>

int main() {
  upsilon::Model model;
  model.Train();
  std::cout << "Training completed." << std::endl;
  model.Predict();
  std::cout << "Prediction completed." << std::endl;
  return 0;
}
