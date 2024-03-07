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
  std::vector<uint32_t> raw_shapes_;
  uint32_t dims_ = 0;
  Eigen::Tensor<float, 3, Eigen::RowMajor> raw_data_;
  Eigen::Tensor<float, 3, Eigen::RowMajor> raw_grads_;

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

  void ZeroGrad() {
    raw_grads_.setZero();
  }

  void Fill(float value) {
    raw_data_.setConstant(value);
    raw_grads_.setZero();
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

    raw_grads_.setZero();
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
    } else if (dims_ == 2) {
      std::cout << raw_data_.chip(0, 0) << '\n';
    }
    else {
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

  void Transform(const std::function<float(float)>& f) {

    for (uint32_t i = 0; i < channels(); i++) {
      for (uint32_t j = 0; j < rows(); j++) {
        for (uint32_t k = 0; k < cols(); k++) {
          raw_data_(i, j, k) = f(raw_data_(i, j, k));
        }
      }
    }
  }

  void Transpose() {
    if (dims_ != 2) {
      throw std::invalid_argument("Transpose only supports 2D tensors");
    }

    if (rows() != cols()) {
      throw std::invalid_argument("Transpose only supports square matrices");
    }
    
    raw_data_ = raw_data_.shuffle(Eigen::array<Eigen::Index, 3>{0, 2, 1});
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

  void Flatten() {
    if (dims_ == 1) {
      return;
    }

    if (dims_ == 2) {
      raw_shapes_ = {raw_shapes_[1] * raw_shapes_[2],1 ,1};
      raw_data_ = raw_data_.reshape(Eigen::array<Eigen::Index, 3>{raw_shapes_[1] * raw_shapes_[2], 1, 1});
      dims_ = 1;
    }

    if (dims_ == 3) {
      raw_shapes_ = {raw_shapes_[0] * raw_shapes_[1] * raw_shapes_[2], 1, 1};
      raw_data_ = raw_data_.reshape(Eigen::array<Eigen::Index, 3>{raw_shapes_[0] * raw_shapes_[1] * raw_shapes_[2], 1, 1});
      dims_ = 1;
    }
  }

  void Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
    if (pads.size() != 4) {
      throw std::invalid_argument("Padding only supports 4 dimensions: up, bottom, left, right.");
    }
    uint32_t pad_rows1 = pads.at(0);  // up
    uint32_t pad_rows2 = pads.at(1);  // bottom
    uint32_t pad_cols1 = pads.at(2);  // left
    uint32_t pad_cols2 = pads.at(3);  // right

    uint32_t new_rows = rows() + pad_rows1 + pad_rows2;
    uint32_t new_cols = cols() + pad_cols1 + pad_cols2;

    Eigen::Tensor<float, 3, Eigen::RowMajor> new_data(channels(), new_rows, new_cols);
    new_data.setConstant(padding_value);

    for (uint32_t i = 0; i < channels(); i++) {
      for (uint32_t j = 0; j < rows(); j++) {
        for (uint32_t k = 0; k < cols(); k++) {
          new_data(i, j + pad_rows1, k + pad_cols1) = raw_data_(i, j, k);
        }
      }
    }

    raw_data_ = new_data;
    raw_shapes_ = {channels(), new_rows, new_cols};
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

  float at(uint32_t i) const {
    return raw_data_.data()[i];
  }

  Tensor<float> mul(const Tensor<float>& other) const {
    if (raw_shapes_ != other.raw_shapes_) {
      throw std::invalid_argument("hadamard product requires same shape");
    }

    Tensor<float> result(raw_shapes_);
    
    for (uint32_t i = 0; i < size(); i++) {
      result.at(i) = at(i) * other.at(i);
    }

    return result;
  }

  int ndim() const {
    return dims_;
  }
};

}  // namespace upsilon
