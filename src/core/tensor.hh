#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace upsilon {

template <typename T = float>
class Tensor {};

template <>
class Tensor<uint8_t> {
  // not implemented yet
};

template <>
class Tensor<float> {
 private:
  std::vector<uint32_t> raw_shapes_;  // Tensor dimensions: channels, rows, cols
  uint32_t dims_;                     // Number of dimensions
  // NOTE: storage order is row, col, channel
  Eigen::Tensor<float, 3, Eigen::RowMajor> raw_data_;      // Tensor data, one matrix per channel
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
      : raw_data_(channels, rows, cols),
        raw_shapes_({channels, rows, cols}) {
    dims_ = 3;
  }

  explicit Tensor(uint32_t size) :  Tensor(1, 1, size) {
    dims_ = 1;
  }

  explicit Tensor(uint32_t rows, uint32_t cols) : Tensor(1, rows, cols) {
    dims_ = 2;
  }

  explicit Tensor(const std::vector<uint32_t> shapes) {
    dims_ = shapes.size();

    switch (dims_) {
      case 3:
        raw_shapes_ = shapes;
        raw_data_.resize(shapes[0], shapes[1], shapes[2]);
        break;
      case 2:
        raw_shapes_ = {1, shapes[0], shapes[1]};
        raw_data_.resize(1, shapes[0], shapes[1]);
        break;
      case 1:
        raw_shapes_ = {1, 1, shapes[0]};
        raw_data_.resize(1, 1, shapes[0]);
        break;
      default:
        throw std::invalid_argument("invalid shape");
    }
  }

  void Fill(float value) {
    raw_data_.setConstant(value);
  }

  void Fill(const std::vector<float>& values) {
    if (values.size() != raw_data_.size()) {
      throw std::invalid_argument("values size does not match tensor size");
    }

    const auto planes = rows() * cols();

    for (uint32_t c = 0; c < channels(); c++) {
      for (uint32_t r = 0; r < rows(); r++) {
        for (uint32_t k = 0; k < cols(); k++) {
          int i = c * planes + r * cols() + k;
          raw_data_(c, r, k) = values.at(i);
        }
      }
    }
  }

  Tensor(const Tensor& tensor) = default;
  Tensor(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(const Tensor& tensor) = default;

   uint32_t size() const {
        return static_cast<uint32_t>(raw_data_.size());
    }

    uint32_t channels() const {
        return raw_shapes_[0];
    }

    uint32_t rows() const {
        return raw_shapes_[1];
    }

    uint32_t cols() const {
        return raw_shapes_[2];
    }

  float& index(uint32_t offset) {
    auto channel = offset / (rows() * cols());
    auto index = offset % (rows() * cols());
    auto row = index / cols();
    auto col = index % cols();
    return raw_data_(channel, row, col);
  }

  void Show() const {
    if (dims_ == 3) {
      for (uint32_t i = 0; i < channels(); i++) {
        std::cout << "Channel " << i << ":\n";
        std::cout << raw_data_.chip(i, 0) << '\n';
      }
    } else {
      std::cout << raw_data_ << '\n';
    }
  }

  std::vector<uint32_t> shape() const {
    switch (dims_) {
      case 1:
        return {raw_shapes_[2]};
      case 2:
        return {raw_shapes_[1], raw_shapes_[2]};
      case 3:
        return {raw_shapes_[0], raw_shapes_[1], raw_shapes_[2]};
      default:
        throw std::invalid_argument("invalid shape");
    }
  }
    
  void Reshape(const std::vector<uint32_t>& shapes) {
      if (shapes.size() != 3) {
          throw std::invalid_argument("Reshape only supports 3 dimensions: channels, rows, cols.");
      }
      dims_ = 3;
      raw_shapes_ = shapes;
      auto new_data = raw_data_.reshape(Eigen::array<Eigen::Index, 3>{shapes[0], shapes[1], shapes[2]});
      raw_data_ = new_data;
  }

  std::vector<float> values() const {
    std::vector<float> values;
    for (uint32_t i = 0; i < channels(); i++) {
      for (uint32_t j = 0; j < rows(); j++) {
        for (uint32_t k = 0; k < cols(); k++) {
          values.push_back(raw_data_(i, j, k));
        }
      }
    }

    return values;
  }

  float& at(uint32_t channel, uint32_t row, uint32_t col) {
    return raw_data_(channel, row, col);
  }

  float& at(uint32_t row, uint32_t col) {
    return raw_data_(0, row, col);
  }

  float& at(uint32_t i) {
    return index(i);
  }
};

}  // namespace upsilon
