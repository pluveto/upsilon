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
  std::vector<uint32_t> raw_shapes_;  // Tensor dimensions, channels, rows, cols
  uint32_t dims_;                     // Number of dimensions
  Eigen::Tensor<float, 3> data_;      // Tensor data, one matrix per channel
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
      : data_(channels, rows, cols),
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

    if (dims_ == 3) {
      raw_shapes_ = shapes;
      data_.resize(shapes[0], shapes[1], shapes[2]);
    } else if (dims_ == 2) {
      raw_shapes_ = {1, shapes[0], shapes[1]};
      data_.resize(1, shapes[0], shapes[1]);
    } else if (dims_ == 1) {
      raw_shapes_ = {1, 1, shapes[0]};
      data_.resize(1, 1, shapes[0]);
    } else {
      throw std::invalid_argument("invalid shape");
    }
  }

  void Fill(float value) {
    data_.setConstant(value);
  }

  Tensor(const Tensor& tensor) = default;
  Tensor(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(const Tensor& tensor) = default;

   uint32_t size() const {
        return static_cast<uint32_t>(data_.size());
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

  float index(uint32_t offset) const {
    // This is a simplified version. You might need a more complex logic to
    // handle multi-channel indexing.
    auto channel = offset / (rows() * cols());
    auto index = offset % (rows() * cols());
    auto row = index / cols();
    auto col = index % cols();
    return data_(channel, row, col);
  }

  float& index(uint32_t offset) {
    auto channel = offset / (rows() * cols());
    auto index = offset % (rows() * cols());
    auto row = index / cols();
    auto col = index % cols();
    return data_(channel, row, col);
  }


  std::vector<uint32_t> shapes() const {
      return raw_shapes_;
  }

  void Show() const {
    std::cout << data_ << '\n';
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
    
  void reshape(const std::vector<uint32_t>& shapes) {
      if (shapes.size() != 3) {
          throw std::invalid_argument("Reshape only supports 3 dimensions: channels, rows, cols.");
      }
      raw_shapes_ = shapes;
      auto new_data = data_.reshape(Eigen::array<Eigen::Index, 3>{shapes[0], shapes[1], shapes[2]});
      data_ = new_data;
  }

  std::vector<float> values(bool row_major = true) const {
    std::vector<float> values;
    for (uint32_t i = 0; i < channels(); i++) {
      for (uint32_t j = 0; j < rows(); j++) {
        for (uint32_t k = 0; k < cols(); k++) {
          values.push_back(data_(i, j, k));
        }
      }
    }

    return values;
  }
};

}  // namespace upsilon
