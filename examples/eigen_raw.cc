#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    std::cout << "2x3 ColMajor ";
    Eigen::Tensor<int, 2> tensor1(2, 3);
    tensor1.setValues({{1, 2, 3}, {4, 5, 6}});
    std::cout << "Data:\n";
    std::cout << tensor1 << std::endl;
    const int* raw_ptr1 = tensor1.data();
    std::cout << "Raw data:\n";
    for (int i = 0; i < tensor1.size(); ++i) {
        std::cout << raw_ptr1[i] << " ";
    }
    std::cout << "\n";

    std::cout << "2x3 RowMajor ";
    Eigen::Tensor<int, 2, Eigen::RowMajor> tensor2(2, 3);
    tensor2.setValues({{1, 2, 3}, {4, 5, 6}});
    std::cout << "Data:\n";
    std::cout << tensor2 << std::endl;
    const int* raw_ptr2 = tensor2.data();
    std::cout << "Raw data:\n";
    for (int i = 0; i < tensor2.size(); ++i) {
        std::cout << raw_ptr2[i] << " ";
    }
    std::cout << "\n";

    std::cout << "2x3x4 RowMajor ";
    Eigen::Tensor<int, 3, Eigen::RowMajor> tensor3(2, 3, 4);
    tensor3.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
                       {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}});
    std::cout << "Data:\n";
    std::cout << tensor3 << std::endl;
    const int* raw_ptr3 = tensor3.data();

    for (int c = 0; c < 2; ++c) {
        std::cout << "Channel " << c << ":\n";
        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 4; ++y) {
                std::cout << tensor3(c, x, y) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Raw data:\n";
    for (int i = 0; i < tensor3.size(); ++i) {
        std::cout << raw_ptr3[i] << " ";
    }
    std::cout << "\n";
    return 0;
}
