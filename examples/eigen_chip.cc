#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    // 创建一个2x3x4的三维张量
    Eigen::Tensor<int, 3> tensor(2, 3, 4);
    tensor.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
                      {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}});

    // 从第二个维度上提取第一个"chip"（即一个3x4的二维张量）
    Eigen::Tensor<int, 2> chip = tensor.chip(0, 0);

    // 打印提取的“chip”
    std::cout << "Extracted chip:\n" << chip << std::endl;

    return 0;
}
