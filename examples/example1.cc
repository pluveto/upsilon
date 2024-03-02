#include <Eigen/Dense>
#include <iostream>

int main() {
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(4, 6);
  std::cout << m << '\n';
  return 0;
}
