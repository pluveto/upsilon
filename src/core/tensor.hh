#pragma once
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace upsilon {

template <typename T = float>
class Tensor {};

template <>
class Tensor<uint8_t> {
  // not implemented yet
};

template <>
class Tensor<float> {
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols)
      : data_(channels, Eigen::MatrixXf(rows, cols)),
        raw_shapes_({channels, rows, cols}) {}

  explicit Tensor(uint32_t size) : Tensor(1, 1, size) {}

  explicit Tensor(uint32_t rows, uint32_t cols) : Tensor(1, rows, cols) {}

  static Tensor create(const std::vector<uint32_t>& shapes) {
    if (shapes.size() == 3) {
      return Tensor(shapes[0], shapes[1], shapes[2]);
    } else if (shapes.size() == 2) {
      return Tensor(shapes[0], shapes[1]);
    } else if (shapes.size() == 1) {
      return Tensor(shapes[0]);
    } else {
      throw std::invalid_argument("invalid shape");
    }
  }

  Tensor(const Tensor& tensor) = default;
  Tensor(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(Tensor&& tensor) noexcept = default;
  Tensor<float>& operator=(const Tensor& tensor) = default;

  uint32_t size() const { return rows() * cols() * channels(); }

  void set_data(const std::vector<Eigen::MatrixXf>& data) { data_ = data; }

  bool empty() const { return data_.empty(); }

  float index(uint32_t offset) const {
    // This is a simplified version. You might need a more complex logic to
    // handle multi-channel indexing.
    auto channel = offset / (rows() * cols());
    auto index = offset % (rows() * cols());
    auto row = index / cols();
    auto col = index % cols();
    return data_[channel](row, col);
  }

  float& index(uint32_t offset) {
    auto channel = offset / (rows() * cols());
    auto index = offset % (rows() * cols());
    auto row = index / cols();
    auto col = index % cols();
    return data_[channel](row, col);
  }

  uint32_t channels() const { return raw_shapes_[0]; }

  uint32_t rows() const { return raw_shapes_[1]; }

  uint32_t cols() const { return raw_shapes_[2]; }

  std::vector<uint32_t> shapes() const { return {channels(), rows(), cols()}; }

  const std::vector<uint32_t>& raw_shapes() const { return raw_shapes_; }

  Eigen::MatrixXf& data(uint32_t channel) { return data_[channel]; }

  const Eigen::MatrixXf& data(uint32_t channel) const { return data_[channel]; }

  float at(uint32_t channel, uint32_t row, uint32_t col) const {
    return data_[channel](row, col);
  }

  float& at(uint32_t channel, uint32_t row, uint32_t col) {
    return data_[channel](row, col);
  }

 private:
  std::vector<uint32_t> raw_shapes_;  // Tensor dimensions, channels, rows, cols
  std::vector<Eigen::MatrixXf> data_;  // Tensor data, one matrix per channel
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

}  // namespace upsilon
