#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace upsilon {

enum class TensorType {
    Scalar,
    Matrix,
    Tensor
};

template<typename T>
using ScalarData = T;

template<typename T>
using MatrixData = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename T>
using TensorData = Eigen::Tensor<T, 3, Eigen::RowMajor>;

template<typename T>
using UnifiedData = std::variant<ScalarData<T>, MatrixData<T>, TensorData<T>>;

template <typename T = float>
class Tensor {};

template <>
class Tensor<uint8_t> {
};

template <>
class Tensor<float> {
private:
  std::vector<uint32_t> raw_shape_;
  TensorType type_;
  UnifiedData<float> raw_data_;

 public:
  TensorType type() const { return type_; }
  
  explicit Tensor(const TensorType type, const std::vector<uint32_t>& shape) : type_(type) {
    if (type == TensorType::Scalar) {
      raw_data_ = ScalarData<float>(0);
      raw_shape_ = {1, 1, 1};
    } else if (type == TensorType::Matrix) {
      raw_data_ = MatrixData<float>(shape[0], shape[1]);
      raw_shape_ = {1, shape[0], shape[1]};
    } else if (type == TensorType::Tensor) {
      raw_data_ = TensorData<float>(shape[0], shape[1], shape[2]);
      raw_shape_ = shape;
    } else {
      throw std::invalid_argument("Invalid tensor type");
    }
  }

  explicit Tensor(const ScalarData<float>& data) : type_(TensorType::Scalar), raw_data_(data) {
    raw_shape_ = {1, 1, 1};
  }

  explicit Tensor(const std::vector<uint32_t>& data, bool row_vector = true) : type_(TensorType::Matrix) {
    if (row_vector) {
      raw_shape_ = {1, 1, static_cast<uint32_t>(data.size())};
      raw_data_ = MatrixData<float>(1, data.size());
    } else {
      raw_shape_ = {1, static_cast<uint32_t>(data.size()), 1,};
      raw_data_ = MatrixData<float>(data.size(), 1);
    }

    for (uint32_t i = 0; i < data.size(); i++) {
      std::get<MatrixData<float>>(raw_data_)(i, 0) = data[i];
    }
  }

  explicit Tensor(const MatrixData<float>& data) : type_(TensorType::Matrix), raw_data_(data) {
    raw_shape_ = {1, static_cast<uint32_t>(data.rows()), static_cast<uint32_t>(data.cols())};
  }

  explicit Tensor(const TensorData<float>& data) : type_(TensorType::Tensor), raw_data_(data) {
    raw_shape_ = {static_cast<uint32_t>(data.dimension(0)), static_cast<uint32_t>(data.dimension(1)), static_cast<uint32_t>(data.dimension(2))};
  }

  UnifiedData<float> data() const {
    return raw_data_;
  }

  void fill(float value) {
    if (type_ == TensorType::Scalar) {
      std::get<ScalarData<float>>(raw_data_) = value;
    } else if (type_ == TensorType::Matrix) {
      std::get<MatrixData<float>>(raw_data_).fill(value);
    } else if (type_ == TensorType::Tensor) {
      std::get<TensorData<float>>(raw_data_).setConstant(value);
    } else {
      throw std::invalid_argument("Invalid tensor type");
    }
  }

  void fill(const std::vector<float>& values) {
    if (values.size() != this->size()) {
      throw std::invalid_argument("values size does not match tensor size");
    }

    if (type_ == TensorType::Scalar) {
      std::get<ScalarData<float>>(raw_data_) = values[0];
    } else if (type_ == TensorType::Matrix) {
      for (uint32_t i = 0; i < values.size(); i++) {
        std::get<MatrixData<float>>(raw_data_).data()[i] = values[i];
      }
    } else if (type_ == TensorType::Tensor) {
      for (uint32_t i = 0; i < values.size(); i++) {
        std::get<TensorData<float>>(raw_data_).data()[i] = values[i];
      }
    } else {
      throw std::invalid_argument("Invalid tensor type");
    }
  }

  uint32_t size() const {
    return raw_shape_[0] * raw_shape_[1] * raw_shape_[2];
  }

  uint32_t channels() const {
    return raw_shape_[0];
  }

  uint32_t rows() const {
    return raw_shape_[1];
  }

  uint32_t cols() const {
      return raw_shape_[2];
  }

  int ndim() const {
    if (this->type() == TensorType::Scalar) {
      return 0;
    } else if (this->type() == TensorType::Matrix) {
      return 2;
    }
  }

  void show() const {
    if (type_ == TensorType::Scalar) {
      std::cout << std::get<ScalarData<float>>(raw_data_) << std::endl;
    } else if (type_ == TensorType::Matrix) {
      std::cout << std::get<MatrixData<float>>(raw_data_) << std::endl;
    } else if (type_ == TensorType::Tensor) {
      for(uint32_t c = 0; c < raw_shape_[0]; c++) {
        std::cout << "Channel " << c << std::endl;
        std::cout << std::get<TensorData<float>>(raw_data_).chip(c, 0) << std::endl;
      }
    }
  }

  std::vector<uint32_t> shape() const {
    if (this->type() == TensorType::Scalar) {
      return {1};
    }

    if (this->type() == TensorType::Matrix) {
      return {raw_shape_[1], raw_shape_[2]};
    }

    return raw_shape_;
  }
    
  void reshape(const std::vector<uint32_t>& new_shape) {
    if (this->type() == TensorType::Scalar) {
      throw std::invalid_argument("Cannot reshape a scalar");
    }

    if (size() != std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<uint32_t>())) {
      throw std::invalid_argument("New shape must have the same number of elements");
    }
    if (type_ == TensorType::Matrix) {
      if (new_shape.size() == 2) {
        using t = Eigen::Map<MatrixData<float>>;
        uint32_t new_rows = new_shape[0];
        uint32_t new_cols = new_shape[1];
        t new_data(std::get<MatrixData<float>>(raw_data_).data(), new_rows, new_cols);
        this->raw_data_ = new_data;
        this->raw_shape_ = {1, new_rows, new_cols};
        return;
      } else if (new_shape.size() == 3) {
        // type conversion to tensor
        auto &matrix = std::get<MatrixData<float>>(raw_data_);
        TensorData<float> new_data(new_shape[0], new_shape[1], new_shape[2]);
        for (size_t i = 0; i < matrix.size(); i++) {
          new_data.data()[i] = matrix.data()[i];
        }
        this->raw_data_ = new_data;
        this->raw_shape_ = new_shape;
        this->type_ = TensorType::Tensor;
        return;
      }

    } else if (type_ == TensorType::Tensor) {
      if (new_shape.size() == 3) {
        auto &tensor = std::get<TensorData<float>>(raw_data_);
        raw_data_ = tensor.reshape(Eigen::array<Eigen::Index, 3>({new_shape[0], new_shape[1], new_shape[2]}));
        this->raw_shape_ = new_shape;
        return;
      } else if (new_shape.size() == 2) {
        // type conversion to matrix
        auto &tensor = std::get<TensorData<float>>(raw_data_);
        MatrixData<float> new_data(new_shape[0], new_shape[1]);
        for (size_t i = 0; i < tensor.size(); i++) {
          new_data.data()[i] = tensor.data()[i];
        }
        this->raw_data_ = new_data;
        this->raw_shape_ = {1, new_shape[0], new_shape[1]};
        this->type_ = TensorType::Matrix;
        return;
      }
    }

    throw std::invalid_argument("Invalid tensor type");
  }

  Tensor<float> apply(const std::function<float(float)>& f) const {
    if (type_ == TensorType::Scalar) {
      Tensor<float> ret({1});
      ret.at(0) = f(std::get<ScalarData<float>>(raw_data_));
      return ret;

    } else if (type_ == TensorType::Matrix) {
      auto& data = std::get<MatrixData<float>>(raw_data_);
      Tensor<float> ret_data(*this);
      for (uint32_t i = 0; i < size(); i++) {
        ret_data.at(i) = f(data.data()[i]);
      }
      return ret_data;

    } else if (type_ == TensorType::Tensor) {
      auto& data = std::get<TensorData<float>>(raw_data_);
      Tensor<float> ret_data(*this);
      for (size_t i = 0; i < data.size(); i++) {
        ret_data.at(i) = f(data.data()[i]);
      }
      return ret_data;

    }
  }

  void transpose() {
    if (type_ == TensorType::Scalar) {
      return;
    }

    if (type_ == TensorType::Matrix) {
      if (raw_shape_[0] != 1) {
        throw std::invalid_argument("Transpose only supports 2D matrix currently");
      }

      std::get<MatrixData<float>>(raw_data_).transposeInPlace();
      raw_shape_ = {raw_shape_[0], raw_shape_[2], raw_shape_[1]};
      return;
    }

    throw std::invalid_argument("Invalid tensor type");
  }

  Tensor<float> transposed() const {
    if (type_ == TensorType::Scalar) {
      return *this;
    }

    if (type_ == TensorType::Matrix) {
      if (raw_shape_[0] != 1) {
        throw std::invalid_argument("Transpose only supports 2D matrix currently");
      }

      Tensor<float> ret(*this);
      ret.transpose();
      return ret;
    }

    throw std::invalid_argument("Invalid tensor type");
  }

  std::vector<float> values() const {
    if (type_ == TensorType::Scalar) {
      return {std::get<ScalarData<float>>(raw_data_)};
    } else if (type_ == TensorType::Matrix) {
      return std::vector<float>(std::get<MatrixData<float>>(raw_data_).data(),
                                std::get<MatrixData<float>>(raw_data_).data() + size());
    } else if (type_ == TensorType::Tensor) {
      return std::vector<float>(std::get<TensorData<float>>(raw_data_).data(),
                                std::get<TensorData<float>>(raw_data_).data() + size());
    }
  }

  void flatten(bool row_vector = true) {
    if (type_ == TensorType::Scalar) {
      return;
    }

    if (type_ == TensorType::Matrix) {
      if (row_vector) {
        this->reshape({1, size()});
      } else {
        this->reshape({size(), 1});
      }

      return;
    }

    if (type_ == TensorType::Tensor) {
      if (row_vector) {
        this->reshape({1, 1, size()});
      } else {
        this->reshape({1, size(), 1});
      }

      return;
    }

    throw std::invalid_argument("Invalid tensor type");
  }

  void padding(const std::vector<uint32_t>& pads,
                            float value) {
    if (pads.size() != 4) {
      throw std::invalid_argument("Padding only supports 4 dimensions: up, bottom, left, right.");
    }

    uint32_t pad_rows1 = pads.at(0);  // up
    uint32_t pad_rows2 = pads.at(1);  // bottom
    uint32_t pad_cols1 = pads.at(2);  // left
    uint32_t pad_cols2 = pads.at(3);  // right

    uint32_t new_rows = rows() + pad_rows1 + pad_rows2;
    uint32_t new_cols = cols() + pad_cols1 + pad_cols2;

    if (new_rows < rows() || new_cols < cols()) {
      throw std::invalid_argument("Invalid padding size");
    }

    if (pad_rows1 == 0 && pad_rows2 == 0 && pad_cols1 == 0 && pad_cols2 == 0) {
      return;
    }
    
    if (this->type_ == TensorType::Matrix) {
      MatrixData<float> new_data(new_rows, new_cols);
      new_data.setConstant(value);

      for (uint32_t i = 0; i < rows(); i++) {
        for (uint32_t j = 0; j < cols(); j++) {
          new_data(i + pad_rows1, j + pad_cols1) = at(i, j);
        }
      }

      this->raw_data_ = new_data;
      this->raw_shape_ = {1, new_rows, new_cols};

      return;
    }

    if (this->type_ == TensorType::Tensor) {
      // padding for each channel
      TensorData<float> new_data(channels(), new_rows, new_cols);
      new_data.setConstant(value);

      for (uint32_t c = 0; c < channels(); c++) {
        for (uint32_t i = 0; i < rows(); i++) {
          for (uint32_t j = 0; j < cols(); j++) {
            new_data(c, i + pad_rows1, j + pad_cols1) = at(c, i, j);
          }
        }
      }

      this->raw_data_ = new_data;
      this->raw_shape_ = {channels(), new_rows, new_cols};

      return;
    }


  }

  float& at(uint32_t i) {
    if (type_ == TensorType::Scalar) {
      if (i == 0) {
        return std::get<ScalarData<float>>(raw_data_);
      } else {
        throw std::invalid_argument("Index out of range");
      }
    }

    if (i >= size()) {
      throw std::invalid_argument("Index out of range");
    }

    if (type_ == TensorType::Matrix) {
      return std::get<MatrixData<float>>(raw_data_).data()[i];
    }
    
    if (type_ == TensorType::Tensor) {
      return std::get<TensorData<float>>(raw_data_).data()[i];
    }

    throw std::invalid_argument("Invalid tensor type");
  }

  float at(uint32_t i) const {
    if (type_ == TensorType::Scalar) {
      if (i == 0) {
        return std::get<ScalarData<float>>(raw_data_);
      } else {
        throw std::invalid_argument("Index out of range");
      }
    }

    if (i >= size()) {
      throw std::invalid_argument("Index out of range");
    }

    return std::get<MatrixData<float>>(raw_data_).data()[i];
  }

  float& at(uint32_t row, uint32_t col) {
    if (type_ == TensorType::Scalar) {
      throw std::invalid_argument("Cannot access element of a scalar");
    }

    if (row >= raw_shape_[1] || col >= raw_shape_[2]) {
      throw std::invalid_argument("Index out of range");
    }

    return std::get<MatrixData<float>>(raw_data_)(row, col);
  }

  float at(uint32_t row, uint32_t col) const {
    if (type_ == TensorType::Scalar) {
      throw std::invalid_argument("Cannot access element of a scalar");
    }

    if (row >= raw_shape_[1] || col >= raw_shape_[2]) {
      throw std::invalid_argument("Index out of range");
    }

    return std::get<MatrixData<float>>(raw_data_)(row, col);
  }

  float& at(uint32_t channel, uint32_t row, uint32_t col) {
    if (type_ == TensorType::Scalar) {
      throw std::invalid_argument("Cannot access element of a scalar");
    }

    if (channel >= raw_shape_[0] || row >= raw_shape_[1] || col >= raw_shape_[2]) {
      throw std::invalid_argument("Index out of range");
    }

    return std::get<TensorData<float>>(raw_data_)(channel, row, col);
  }

  float at(uint32_t channel, uint32_t row, uint32_t col) const {
    if (type_ == TensorType::Scalar) {
      throw std::invalid_argument("Cannot access element of a scalar");
    }

    if (channel >= raw_shape_[0] || row >= raw_shape_[1] || col >= raw_shape_[2]) {
      throw std::invalid_argument("Index out of range");
    }

    return std::get<TensorData<float>>(raw_data_)(channel, row, col);
  }

  Tensor<float> mul(const Tensor<float>& other) const {
    if (raw_shape_ != other.raw_shape_) {
      throw std::invalid_argument("Hadamard product requires same shape");
    }

    if (type_ == TensorType::Scalar && other.type() == TensorType::Scalar) {
      return Tensor<float>(std::get<ScalarData<float>>(raw_data_) * std::get<ScalarData<float>>(other.raw_data_));
    }

    Tensor<float> result(raw_shape_);
    
    for (uint32_t i = 0; i < size(); i++) {
      result.at(i) = at(i) * other.at(i);
    }

    return result;
  }

  Tensor<float> add(const Tensor<float>& other) const {
    if (raw_shape_ != other.raw_shape_) {
      throw std::invalid_argument("Addition requires same shape");
    }

    if (type_ == TensorType::Scalar && other.type() == TensorType::Scalar) {
      return Tensor<float>(std::get<ScalarData<float>>(raw_data_) + std::get<ScalarData<float>>(other.raw_data_));
    }

    Tensor<float> result(this->shape());
    
    for (uint32_t i = 0; i < size(); i++) {
      result.at(i) = at(i) + other.at(i);
    }

    return result;
  }

  Tensor<float> sub(const Tensor<float>& other) const {
    if (raw_shape_ != other.raw_shape_) {
      throw std::invalid_argument("Subtraction requires same shape");
    }

    Tensor<float> result(this->shape());
    
    for (uint32_t i = 0; i < size(); i++) {
      result.at(i) = at(i) - other.at(i);
    }

    return result;
  }

  Tensor<float> div(const Tensor<float>& other) const {
    if (raw_shape_ != other.raw_shape_) {
      throw std::invalid_argument("Division requires same shape");
    }

    Tensor<float> result(this->shape());
    
    for (uint32_t i = 0; i < size(); i++) {
      result.at(i) = at(i) / other.at(i);
    }

    return result;
  }

  Tensor<float> matmul(const Tensor<float>& other) const {
    if (this->type() != TensorType::Matrix || other.type() != TensorType::Matrix) {
      throw std::invalid_argument("Matrix multiplication requires 2D matrix");
    }

    if (this->cols() != other.rows()) {
      throw std::invalid_argument("Matrix multiplication requires the number of columns of the first matrix to be equal to the number of rows of the second matrix");
    }

    MatrixData<float> result = std::get<MatrixData<float>>(raw_data_) * std::get<MatrixData<float>>(other.raw_data_);
    return Tensor<float>(result);
  }

  Tensor<float> inv() const {
    if (this->type() != TensorType::Matrix) {
      throw std::invalid_argument("Matrix inversion requires 2D matrix");
    }

    return Tensor<float>(std::get<MatrixData<float>>(raw_data_).inverse());
  }

  Tensor<float> square() const {
    const auto sqrt_fn = [](float x) { return x * x; };
    return this->apply(sqrt_fn);
  }

  Tensor<float> pow(float exponent) const {
    const auto pow_fn = [exponent](float x) { return std::pow(x, exponent); };
    return this->apply(pow_fn);
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<float>& obj) {
    if (obj.type() == TensorType::Scalar) {
      os << std::get<ScalarData<float>>(obj.data());
    } else if (obj.type() == TensorType::Matrix) {
      os << std::get<MatrixData<float>>(obj.data());
    } else if (obj.type() == TensorType::Tensor) {
      for(uint32_t c = 0; c < obj.shape()[0]; c++) {
        os << "Channel " << c << std::endl;
        os << std::get<TensorData<float>>(obj.data()).chip(c, 0) << std::endl;
      }
    }

    return os;
  }

  static Tensor<float> zeros_like(const Tensor<float>& other) {
    if (other.type() == TensorType::Scalar) {
      return Tensor<float>(0.0f);
    }

    if (other.type() == TensorType::Matrix) {
      return Tensor<float>(TensorType::Matrix, {other.rows(), other.cols()});
    }

    if (other.type() == TensorType::Tensor) {
      return Tensor<float>(TensorType::Tensor, {other.channels(), other.rows(), other.cols()});
    }
  }

};

}  // namespace upsilon
