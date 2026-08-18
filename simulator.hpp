#ifndef SIMULATION_HPP
#define SIMULATION_HPP
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <sys/types.h>
#include <tuple>
#include <vector>

namespace sjtu {

class Matrix;

class Rater;

class GpuSimulator;

class MatrixMemoryAllocator;

enum class InstructionType : int {
  kMatAdd,
  kMatSub,
  kMatmul,
  kMatExp,
  kMatDiv,
  kAddBias,
  kMulNum,
  kConcat,
  kGetRow,
  kGetColumn,
  kToGpuHbm,
  kToSharedMem,
  kRelease,
  kSum,
  kTranspose,
  kReshape,
  kCopy,
};
enum Position { kInGpuHbm, kInSharedMemory, kReleased };
class Matrix {
public:
  Matrix() = default;

  Matrix(size_t row_num, size_t column_num);

  Matrix(size_t row_num, size_t column_num, const std::vector<float> &data,
         GpuSimulator &gpu_sim);

  ~Matrix() = default;

  /* Set every element to 0.*/
  void Zero() { std::fill(data_.begin(), data_.end(), 0.0f); }

  /* Give every element a random value.*/
  void Rand() {
    for (size_t i = 0; i < GetSize(); ++i) {
      data_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  }

  Position GetPosition() const { return position_; }

  void PrintShape() const {
    std::cerr << "(" << row_num_ << ", " << column_num_ << ")" << std::endl;
  }

  void Print() const {
    std::cerr << "Matrix data: " << std::endl;
    for (size_t i = 0; i < row_num_; ++i) {
      for (size_t j = 0; j < column_num_; ++j) {
        std::cerr << std::fixed << std::setprecision(1)
                  << data_[GetDataIndex(i, j)] << " ";
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
  }

  size_t GetSize() const { return column_num_ * row_num_; }

  size_t GetColumnNum() const { return column_num_; }

  size_t GetRowNum() const { return row_num_; }

  size_t GetDataIndex(size_t row_index, size_t column_index) const {
    return row_index * column_num_ + column_index;
  }

  friend GpuSimulator;

  friend Rater;

private:
  void Transpose() {
    std::vector<float> transposed_data(GetSize());
    for (size_t i = 0; i < row_num_; ++i) {
      for (size_t j = 0; j < column_num_; ++j) {
        transposed_data[j * row_num_ + i] = data_[i * column_num_ + j];
      }
    }
    data_ = std::move(transposed_data);
    std::swap(row_num_, column_num_);
  }

  Matrix(const Matrix &other) = default;

  void Reshape(size_t row_num, size_t column_num) {
    if (row_num * column_num != GetSize()) {
      std::cerr << "New shape does not match the number of elements."
                << std::endl;
      return;
    }
    row_num_ = row_num;
    column_num_ = column_num;
    data_.resize(GetSize());
    std::cerr << "Matrix reshaped to: ";
    PrintShape();
  }

  static Matrix MatMul(const Matrix *matrix1, const Matrix *matrix2,
                       GpuSimulator &gpu_sim);

  static Matrix MatDiv(const Matrix *matrix1, const Matrix *divisor,
                       GpuSimulator &gpu_sim);

  static Matrix MatExp(const Matrix *matrix, GpuSimulator &gpu_sim);

  static Matrix MatAdd(const Matrix *matrix1, const Matrix *matrix2,
                       GpuSimulator &gpu_sim);

  static Matrix MatSub(const Matrix *matrix1, const Matrix *matrix2,
                       GpuSimulator &gpu_sim);

  static Matrix AddBias(const Matrix *matrix1, const Matrix *bias,
                        GpuSimulator &gpu_sim);

  static Matrix MulNum(const Matrix *matrix1, const Matrix *factor,
                       GpuSimulator &gpu_sim);

  static Matrix Concat(const Matrix *matrix1, const Matrix *matrix2,
                       size_t axis, GpuSimulator &gpu_sim);

  static Matrix GetRow(const Matrix *matrix, size_t row_index,
                       GpuSimulator &gpu_sim);

  static Matrix GetColumn(const Matrix *matrix, size_t column_index,
                          GpuSimulator &gpu_sim);

  static Matrix Sum(const Matrix *matrix, GpuSimulator &gpu_sim);

  Matrix &operator=(const Matrix &other) {
    if (this == &other) {
      return *this; // Handle self-assignment
    }

    if ((this->row_num_ != other.row_num_ ||
         this->column_num_ != other.column_num_) &&
        this->GetSize() != 0) {
      std::cerr << "Matrix dimensions must match for assignment. The original "
                   "matrix would be overwrite."
                << this->GetSize() << std::endl;
    }
    row_num_ = other.row_num_;
    column_num_ = other.column_num_;
    data_ = other.data_;
    position_ = other.position_;
    return *this;
  }

  std::vector<float> data_;
  size_t row_num_ = 0;
  size_t column_num_ = 0;
  Position position_ = kReleased;
};

class GpuSimulator {

public:
  /* ************************ Memory Instruction ************************ */

  /*!
   * \brief Move a matrix to GPU HBM.
   * \param matrix The matrix to be moved.
   * \note This operation will cost 300 size(matrix) cycles.
   */
  void MoveMatrixToGpuHbm(Matrix *matrix);

  /*!
   * \brief Move a matrix to shared memory.
   * \param matrix The matrix to be moved.
   * \note This operation will cost 300 size(matrix) cycles.
   */
  void MoveMatrixToSharedMem(Matrix *matrix);

  /* ******************* Sram Calculation Instruction ******************* */

  /*!
   * \brief Add two matrices and store the result in a third matrix.
   * \param lhs The left-hand side matrix.
   * \param rhs The right-hand side matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost size(lhs) cycles.
   */
  void MatAdd(Matrix *lhs, Matrix *rhs, Matrix *result);

  /*!
   * \brief Subtract two matrices and store the result in a third matrix.
   * \param lhs The left-hand side matrix.
   * \param rhs The right-hand side matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost size(lhs) cycles.
   */
  void MatSub(Matrix *lhs, Matrix *rhs, Matrix *result);

  /*!
   * \brief Multiply two matrices and store the result in a third matrix.
   * \param lhs The left-hand side matrix.
   * \param rhs The right-hand side matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost 5 * size(lhs) * size(rhs) cycles.
   */
  void MatMul(Matrix *lhs, Matrix *rhs, Matrix *result);

  /*!
   * \brief Perform matrix exponentiation.
   * \param matrix The matrix to be exponentiated.
   * \param result The matrix to store the result.
   * \note This operation will cost 30 size(matrix) cycles.
   */
  void MatExp(Matrix *matrix, Matrix *result);

  /*!
   * \brief Perform a division operation on two matrices.
   * \param matrix The matrix to be divided.
   * \param divisor The matrix to divide by. It must be a 1x1 matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost 15 size(matrix) cycles.
   */
  void MatDiv(Matrix *matrix, Matrix *divisor, Matrix *result);

  /*!
   * \brief Compute the sum of all elements in a matrix.
   * \param matrix The matrix to sum.
   * \param result The matrix to store the result. It will be a 1x1 matrix.
   * \note This operation will cost size(matrix) cycles.
   */
  void Sum(Matrix *matrix, Matrix *result);

  /*!
   * \brief Add a bias to a matrix.
   * \param matrix The matrix to which the bias will be added.
   * \param bias The bias value to add. It's a 1x1 matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost size(matrix) cycles.
   */
  void MatAddBias(Matrix *matrix, Matrix *bias, Matrix *result);

  /*!
   * \brief Multiply a matrix by a scalar.
   * \param matrix The matrix to be multiplied.
   * \param factor The scalar factor. It's a 1x1 matrix.
   * \param result The matrix to store the result.
   * \note This operation will cost 4 * size(matrix) cycles.
   */
  void MatMulNum(Matrix *matrix, Matrix *factor, Matrix *result);

  /* ****************** General Calculation Instruction ****************** */

  /*!
   * \brief Concatenate two matrices along a specified axis.
   * \param matrix1 The first matrix to concatenate.
   * \param matrix2 The second matrix to concatenate.
   * \param result The matrix to store the result.
   * \param axis The axis along which to concatenate.
   * \param position Where to do the concatenation.
   * \note This operation will cost size(matrix1) + size(matrix2) cycles(Sram),
   * 25 x (size(matrix1) + size(matrix2)) cycles(HBM).
   * \note The axis must be either 0 or 1.
   * \note If axis is 0, the matrices are concatenated vertically (along rows);
   * if axis is 1, they are concatenated horizontally (along columns).
   */
  void Concat(Matrix *matrix1, Matrix *matrix2, Matrix *result, size_t axis,
              Position position);

  /*!
   * \brief Get a specific row from a matrix.
   * \param matrix The matrix from which to get the row.
   * \param row_index The index of the row to get.
   * \param result The matrix to store the result.
   * \param position Where to do the operation.
   * \note This operation will cost size(column_counts) cycles(Sram),
   * 25 x size(column_counts) cycles(HBM).
   */
  void GetRow(Matrix *matrix, size_t row_index, Matrix *result,
              Position position);

  /*!
   * \brief Get a specific column from a matrix.
   * \param matrix The matrix from which to get the column.
   * \param column_index The index of the column to get.
   * \param result The matrix to store the result.
    * \param position Where to do the operation.

   * \note This operation will cost size(row_counts) cycles.
   * 25 x size(row_counts) cycles(HBM).
   */
  void GetColumn(Matrix *matrix, size_t column_index, Matrix *result,
                 Position position);

  /*!
   * \brief Reshape a matrix to a new shape.
   * \param matrix The matrix to be reshaped.
   * \param new_row_num The new number of rows.
   * \note This operation will cost 1 cycle.
   */
  void Reshape(Matrix *matrix, size_t new_row_num);

  /*!
   * \brief Release a matrix from GPU memory.
   * \param matrix The matrix to be released.
   * \note Once released, the matrix cannot be used in further calculations.
   * \note This operation will cost 10 cycles.
   */
  void ReleaseMatrix(Matrix *matrix);

  /*!
   * \brief Transpose a matrix.
   * \param matrix The matrix to be transposed.
   * \param position Where to do the operation.
   * \note If the matrix is in HBM, then it will take 25 * size(matrix) cycles.
   * \note If the matrix is in SRAM, then it will take size(matrix) cycles.
   */
  void Transpose(Matrix *matrix, Position position);

  /*!
   * \brief Copy a matrix to another matrix.
   * \param src The source matrix to copy from.
   * \param dest The destination matrix to copy to.
   * \param position Where to do the copy operation.
   * \note If the destination matrix is in HBM, then it will take
   * 25 * size(src) cycles.
   * \note If the destination matrix is in SRAM, then it will take
   * size(src) cycles.
   */
  void Copy(Matrix *src, Matrix *dest, Position position);

  /* ****************************** Others ****************************** */

  friend Matrix;

  friend Rater;

  GpuSimulator() = default;

  GpuSimulator(const GpuSimulator &) = delete;

  GpuSimulator(GpuSimulator &&) = delete;

  GpuSimulator &operator=(const GpuSimulator &) = delete;

  ~GpuSimulator() = default;

  void Run(bool debug_print = false,
           const MatrixMemoryAllocator *matrix_memory_allocator = nullptr) {
    while (Advance(debug_print, matrix_memory_allocator))
      ;
  }

private:
  using time_stamp = long long;
  using CalculationInstruction =
      std::tuple<time_stamp, InstructionType, Matrix *, Matrix *, size_t,
                 Matrix *, Position>;
  using IoInstruction = std::tuple<time_stamp, InstructionType, Matrix *>;
  std::queue<IoInstruction> io_queue_;
  std::queue<CalculationInstruction> calculate_queue_;
  long long gpu_sim_cycle_ = 0;
  long long gpu_hbm_used_ = 0;
  long long shared_mem_used_ = 0;
  long long max_gpu_hbm_size_ = 0;
  long long max_shared_mem_size_ = 0;

  /*!
   \brief Pick the next instruction to execute from the queue.
   \return True if an instruction was picked, false if the queue is empty.
  */
  bool Advance(bool debug_print = false,
               const MatrixMemoryAllocator *matrix_memory_allocator = nullptr);

  inline void DoCalc(CalculationInstruction instruction);

  inline void DoIO(IoInstruction instruction);

  uint64_t GetCurrentCycle() const { return gpu_sim_cycle_; }

  void UpdateTimeOfInstructions() const;

  void
  PrintInstruction(const CalculationInstruction &instruction,
                   const MatrixMemoryAllocator &matrix_memory_allocator) const;

  void
  PrintInstruction(const IoInstruction &instruction,
                   const MatrixMemoryAllocator &matrix_memory_allocator) const;
};

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator);

class Rater {
private:
  std::vector<Matrix *> keys_;
  std::vector<Matrix *> values_;
  std::vector<Matrix *> queries_;
  std::vector<Matrix *> answers_;
  std::vector<int> errors_;
  int current_query_index_ = 0;
  bool next_query_available_ = true;

public:
  Rater() = delete;

  Rater(const Rater &) = delete;

  Rater &operator=(const Rater &) = delete;

  Rater(Rater &&) = delete;

  ~Rater() = default;

  Rater(const std::vector<Matrix *> &keys, const std::vector<Matrix *> &values,
        const std::vector<Matrix *> &queries,
        const std::vector<Matrix *> &answers)
      : keys_(keys), values_(values), queries_(queries), answers_(answers) {}

  Matrix *GetNextQuery();

  void CommitAnswer(Matrix &answer);

  int GetErrorsCount() const {
    int result = 0;
    for (const auto &error : errors_) {
      result += error;
    }
    return result;
  }

  void PrintResult(const GpuSimulator &) const;

  friend void Test(Rater &rater, GpuSimulator &gpu_sim,
                   MatrixMemoryAllocator &matrix_memory_allocator);

  void Check();
};

class MatrixMemoryAllocator {
private:
  std::map<Matrix *, std::string> allocated_matrices_;

public:
  MatrixMemoryAllocator() = default;

  // ~MatrixMemoryAllocator() {
  //   for (auto matrix_pair : allocated_matrices_) {
  //     delete matrix_pair.first;
  //   }
  // }

  Matrix *Allocate(std::string name = "") {
    Matrix *matrix = new Matrix();
    allocated_matrices_[matrix] = name;
    return matrix;
  }

  std::string GetMatrixName(Matrix *matrix) const {
    auto it = allocated_matrices_.find(matrix);
    if (it != allocated_matrices_.end()) {
      return it->second;
    }
    return "Unknown Matrix";
  }

  void Bind(Matrix *matrix, const std::string &name) {
    allocated_matrices_[matrix] = name;
  }
};

Matrix Matrix::GetColumn(const Matrix *matrix, size_t column_index,
                         GpuSimulator &gpu_sim) {
  if (column_index >= matrix->GetColumnNum()) {
    std::cerr << "Column index out of bounds." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(matrix->GetRowNum(), 1);
  for (size_t i = 0; i < matrix->GetRowNum(); ++i) {
    result.data_[i] = matrix->data_[i * matrix->GetColumnNum() + column_index];
  }
  result.position_ = matrix->position_;
  if (result.position_ == kInGpuHbm) {
    gpu_sim.gpu_hbm_used_ += result.GetSize();
  } else {
    gpu_sim.shared_mem_used_ += result.GetSize();
  }
  return result;
}

Matrix Matrix::GetRow(const Matrix *matrix, size_t row_index,
                      GpuSimulator &gpu_sim) {
  if (row_index >= matrix->GetRowNum()) {
    std::cerr << "Row index out of bounds." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(1, matrix->GetColumnNum());
  for (size_t j = 0; j < matrix->GetColumnNum(); ++j) {
    result.data_[j] = matrix->data_[row_index * matrix->GetColumnNum() + j];
  }
  result.position_ = matrix->position_;
  if (result.position_ == kInGpuHbm) {
    gpu_sim.gpu_hbm_used_ += result.GetSize();
  } else {
    gpu_sim.shared_mem_used_ += result.GetSize();
  }
  return result;
}

Matrix Matrix::Concat(const Matrix *matrix1, const Matrix *matrix2, size_t axis,
                      GpuSimulator &gpu_sim) {
  if (matrix1->GetPosition() != matrix2->GetPosition()) {
    std::cerr
        << "Matrices must be in the same memory position for concatenation."
        << std::endl;
    exit(EXIT_FAILURE);
  }
  if (axis == 0) { // Concatenate vertically
    if (matrix1->GetColumnNum() != matrix2->GetColumnNum()) {
      std::cerr << "Matrix concatenation dimension mismatch." << std::endl;
      exit(EXIT_FAILURE);
    }
    Matrix result(matrix1->GetRowNum() + matrix2->GetRowNum(),
                  matrix1->GetColumnNum());
    for (size_t i = 0; i < matrix1->GetRowNum(); ++i) {
      for (size_t j = 0; j < matrix1->GetColumnNum(); ++j) {
        result.data_[i * result.GetColumnNum() + j] =
            matrix1->data_[i * matrix1->GetColumnNum() + j];
      }
    }
    for (size_t i = 0; i < matrix2->GetRowNum(); ++i) {
      for (size_t j = 0; j < matrix2->GetColumnNum(); ++j) {
        result.data_[(i + matrix1->GetRowNum()) * result.GetColumnNum() + j] =
            matrix2->data_[i * matrix2->GetColumnNum() + j];
      }
    }
    result.position_ = matrix1->GetPosition();
    if (result.position_ == kInGpuHbm)
      gpu_sim.gpu_hbm_used_ += result.GetSize();
    else {
      gpu_sim.shared_mem_used_ += result.GetSize();
    }
    return result;
  } else if (axis == 1) { // Concatenate horizontally
    if (matrix1->GetRowNum() != matrix2->GetRowNum()) {
      std::cerr << "Matrix concatenation dimension mismatch." << std::endl;
      exit(EXIT_FAILURE);
    }
    Matrix result(matrix1->GetRowNum(),
                  matrix1->GetColumnNum() + matrix2->GetColumnNum());
    for (size_t i = 0; i < matrix1->GetRowNum(); ++i) {
      for (size_t j = 0; j < matrix1->GetColumnNum(); ++j) {
        result.data_[i * result.GetColumnNum() + j] =
            matrix1->data_[i * matrix1->GetColumnNum() + j];
      }
      for (size_t j = 0; j < matrix2->GetColumnNum(); ++j) {
        result
            .data_[i * result.GetColumnNum() + (j + matrix1->GetColumnNum())] =
            matrix2->data_[i * matrix2->GetColumnNum() + j];
      }
    }
    result.position_ = matrix1->GetPosition();
    if (result.position_ == kInGpuHbm)
      gpu_sim.gpu_hbm_used_ += result.GetSize();
    else {
      gpu_sim.shared_mem_used_ += result.GetSize();
    }
    return result;
  } else {
    std::cerr << "Invalid axis for concatenation." << std::endl;
    exit(EXIT_FAILURE);
  }
}

Matrix Matrix::MulNum(const Matrix *matrix1, const Matrix *factor,
                      GpuSimulator &gpu_sim) {
  if (factor->GetSize() != 1) {
    std::cerr << "Factor must be a 1x1 matrix." << std::endl;
    exit(EXIT_FAILURE);
  }
  auto factor_value = factor->data_[0];
  Matrix result(matrix1->GetRowNum(), matrix1->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetSize(); ++i) {
    result.data_[i] = matrix1->data_[i] * factor_value;
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::AddBias(const Matrix *matrix1, const Matrix *bias,
                       GpuSimulator &gpu_sim) {
  if (bias->GetSize() != 1) {
    std::cerr << "Bias must be a 1x1 matrix." << std::endl;
    exit(EXIT_FAILURE);
  }
  auto bias_value = bias->data_[0];
  Matrix result(matrix1->GetRowNum(), matrix1->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetSize(); ++i) {
    result.data_[i] = matrix1->data_[i] + bias_value;
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::MatSub(const Matrix *matrix1, const Matrix *matrix2,
                      GpuSimulator &gpu_sim) {
  if (matrix1->GetRowNum() != matrix2->GetRowNum() ||
      matrix1->GetColumnNum() != matrix2->GetColumnNum()) {
    std::cerr << "Matrix subtraction dimension mismatch." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(matrix1->GetRowNum(), matrix1->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetSize(); ++i) {
    result.data_[i] = matrix1->data_[i] - matrix2->data_[i];
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::MatAdd(const Matrix *matrix1, const Matrix *matrix2,
                      GpuSimulator &gpu_sim) {
  if (matrix1->GetRowNum() != matrix2->GetRowNum() ||
      matrix1->GetColumnNum() != matrix2->GetColumnNum()) {
    std::cerr << "Matrix addition dimension mismatch." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(matrix1->GetRowNum(), matrix1->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetSize(); ++i) {
    result.data_[i] = matrix1->data_[i] + matrix2->data_[i];
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix::Matrix(size_t row_num, size_t column_num)
    : row_num_(row_num), column_num_(column_num), position_(kInGpuHbm) {
  if (row_num <= 0 || column_num <= 0) {
    std::cerr << "Matrix dimensions must be positive. Set the matrix to "
                 "empty matrix."
              << std::endl;
    row_num_ = 0;
    column_num_ = 0;
  }
  data_.resize(GetSize());
  position_ = kReleased;
}

Matrix::Matrix(size_t row_num, size_t column_num,
               const std::vector<float> &data, GpuSimulator &gpu_sim)
    : data_(data), row_num_(row_num), column_num_(column_num),
      position_(kInGpuHbm) {
  if (row_num <= 0 || column_num <= 0) {
    std::cerr << "Matrix dimensions must be positive. Set the matrix to "
                 "empty matrix."
              << std::endl;
    row_num_ = 0;
    column_num_ = 0;
  }
  if (data.size() != row_num * column_num) {
    std::cerr << "Data size does not match the matrix dimensions. Set the "
                 "matrix to empty matrix."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  gpu_sim.gpu_hbm_used_ += GetSize();
}

Matrix Matrix::MatMul(const Matrix *matrix1, const Matrix *matrix2,
                      GpuSimulator &gpu_sim) {
  if (matrix1->GetColumnNum() != matrix2->GetRowNum()) {
    std::cerr << "Matrix multiplication dimension mismatch." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(matrix1->GetRowNum(), matrix2->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetRowNum(); ++i) {
    for (size_t j = 0; j < matrix2->GetColumnNum(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < matrix1->GetColumnNum(); ++k) {
        sum += matrix1->data_[i * matrix1->GetColumnNum() + k] *
               matrix2->data_[k * matrix2->GetColumnNum() + j];
      }
      result.data_[i * result.GetColumnNum() + j] = sum;
    }
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::MatDiv(const Matrix *matrix1, const Matrix *divisor,
                      GpuSimulator &gpu_sim) {
  if (divisor->GetSize() != 1) {
    std::cerr << "Matrix division dimension mismatch." << std::endl;
    exit(EXIT_FAILURE);
  }
  Matrix result(matrix1->GetRowNum(), matrix1->GetColumnNum());
  for (size_t i = 0; i < matrix1->GetSize(); ++i) {
    if (divisor->data_[0] == 0.0f) {
      std::cerr << "Division by zero encountered." << std::endl;
      exit(EXIT_FAILURE);
    }
    result.data_[i] = matrix1->data_[i] / divisor->data_[0];
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::MatExp(const Matrix *matrix, GpuSimulator &gpu_sim) {
  Matrix result(matrix->GetRowNum(), matrix->GetColumnNum());
  for (size_t i = 0; i < matrix->GetSize(); ++i) {
    result.data_[i] = exp(matrix->data_[i]);
  }
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

Matrix Matrix::Sum(const Matrix *matrix, GpuSimulator &gpu_sim) {
  Matrix result(1, 1);
  float sum = 0.0f;
  for (const auto &value : matrix->data_) {
    sum += value;
  }
  result.data_[0] = sum;
  result.position_ = kInSharedMemory;
  gpu_sim.shared_mem_used_ += result.GetSize();
  return result;
}

void GpuSimulator::MoveMatrixToGpuHbm(Matrix *matrix) {
  io_queue_.push({-1, InstructionType::kToGpuHbm, matrix});
}

void GpuSimulator::MoveMatrixToSharedMem(Matrix *matrix) {
  io_queue_.push({-1, InstructionType::kToSharedMem, matrix});
}

void GpuSimulator::ReleaseMatrix(Matrix *matrix) {
  calculate_queue_.push({-1, InstructionType::kRelease, matrix, nullptr, 0,
                         nullptr, kInSharedMemory});
}

void GpuSimulator::MatAdd(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kMatAdd, matrix1, matrix2, 0,
                         result, kInSharedMemory});
}

void GpuSimulator::MatSub(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kMatSub, matrix1, matrix2, 0,
                         result, kInSharedMemory});
}

void GpuSimulator::MatMul(Matrix *matrix1, Matrix *matrix2, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kMatmul, matrix1, matrix2, 0,
                         result, kInSharedMemory});
}

void GpuSimulator::MatDiv(Matrix *matrix1, Matrix *divisor, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kMatDiv, matrix1, divisor, 0,
                         result, kInSharedMemory});
}

void GpuSimulator::MatExp(Matrix *matrix, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kMatExp, matrix, nullptr, 0,
                         result, kInSharedMemory});
}

void GpuSimulator::Copy(Matrix *src, Matrix *dest, Position position) {
  calculate_queue_.push(
      {-1, InstructionType::kCopy, src, nullptr, 0, dest, position});
}

void GpuSimulator::GetColumn(Matrix *matrix, size_t column_index,
                             Matrix *result, Position position) {
  calculate_queue_.push({-1, InstructionType::kGetColumn, matrix, nullptr,
                         column_index, result, position});
}

void GpuSimulator::GetRow(Matrix *matrix, size_t row_index, Matrix *result,
                          Position position) {
  calculate_queue_.push({-1, InstructionType::kGetRow, matrix, nullptr,
                         row_index, result, position});
}

void GpuSimulator::Concat(Matrix *matrix1, Matrix *matrix2, Matrix *result,
                          size_t axis, Position position) {
  calculate_queue_.push(
      {-1, InstructionType::kConcat, matrix1, matrix2, axis, result, position});
}

void GpuSimulator::Sum(Matrix *matrix, Matrix *result) {
  calculate_queue_.push({-1, InstructionType::kSum, matrix, nullptr, -1, result,
                         kInSharedMemory});
}

void GpuSimulator::Transpose(Matrix *matrix, Position position) {
  calculate_queue_.push(
      {-1, InstructionType::kTranspose, matrix, nullptr, 0, nullptr, position});
}

void GpuSimulator::Reshape(Matrix *matrix, size_t new_row_num) {
  calculate_queue_.push({-1, InstructionType::kReshape, matrix, nullptr,
                         new_row_num, nullptr, kInSharedMemory});
}

void GpuSimulator::UpdateTimeOfInstructions() const {
  // Update the time stamps of the instructions in the queues.
  if (!calculate_queue_.empty()) {
    auto &front_instruction =
        const_cast<decltype(calculate_queue_)::value_type &>(
            calculate_queue_.front());
    if (std::get<0>(front_instruction) < 0) {
      auto instruction_type = std::get<1>(front_instruction);
      auto matrix1 = std::get<2>(front_instruction);
      auto matrix2 = std::get<3>(front_instruction);
      switch (instruction_type) {
      case InstructionType::kConcat: {
        if (matrix1->GetPosition() != matrix2->GetPosition() ||
            matrix1->GetPosition() == Position::kReleased ||
            matrix1->GetPosition() != std::get<6>(front_instruction)) {
          break;
        }
        std::get<0>(front_instruction) =
            (matrix1->GetSize() + matrix2->GetSize()) *
            (matrix1->GetPosition() == Position::kInSharedMemory ? 1 : 25);
        break;
      }
      case InstructionType::kMatAdd:
      case InstructionType::kMatSub:
      case InstructionType::kMatmul:
      case InstructionType::kMatDiv:
      case InstructionType::kAddBias:
      case InstructionType::kMulNum: {
        if (matrix1->GetPosition() != Position::kInSharedMemory ||
            matrix2->GetPosition() != Position::kInSharedMemory) {
          break;
        }
        if (instruction_type == InstructionType::kMatAdd ||
            instruction_type == InstructionType::kMatSub) {
          std::get<0>(front_instruction) = matrix1->GetSize();
        } else if (instruction_type == InstructionType::kMatmul) {
          std::get<0>(front_instruction) =
              5 * matrix1->GetSize() * matrix2->GetSize();
        } else if (instruction_type == InstructionType::kMatDiv) {
          std::get<0>(front_instruction) = 16 * matrix1->GetSize();
        } else if (instruction_type == InstructionType::kAddBias) {
          std::get<0>(front_instruction) = matrix1->GetSize();
        } else {
          std::get<0>(front_instruction) = 4 * matrix1->GetSize();
        }
        break;
      }
      case InstructionType::kMatExp:
      case InstructionType::kSum: {
        if (matrix1->GetPosition() != Position::kInSharedMemory) {
          break;
        }
        if (instruction_type == InstructionType::kMatExp) {
          std::get<0>(front_instruction) = 30 * matrix1->GetSize();
        } else if (instruction_type == InstructionType::kSum) {
          std::get<0>(front_instruction) = matrix1->GetSize();
        }
        break;
      }
      case InstructionType::kGetRow:
      case InstructionType::kGetColumn:
      case InstructionType::kReshape:
      case InstructionType::kTranspose:
      case InstructionType::kCopy: {
        if (matrix1->GetPosition() == Position::kReleased ||
            matrix1->GetPosition() != std::get<6>(front_instruction)) {
          break;
        }
        if (instruction_type == InstructionType::kGetRow) {
          std::get<0>(front_instruction) =
              matrix1->GetColumnNum() *
              (matrix1->GetPosition() == Position::kInSharedMemory ? 1 : 25);
        } else if (instruction_type == InstructionType::kGetColumn) {
          std::get<0>(front_instruction) =
              matrix1->GetRowNum() *
              (matrix1->GetPosition() == Position::kInSharedMemory ? 1 : 25);
        } else if (instruction_type == InstructionType::kReshape) {
          std::get<0>(front_instruction) = 1;
        } else {
          std::get<0>(front_instruction) =
              matrix1->GetSize() *
              (matrix1->GetPosition() == Position::kInSharedMemory ? 1 : 25);
        }
        break;
      }
      case InstructionType::kRelease: {
        if (matrix1->GetPosition() == Position::kReleased) {
          break;
        }
        std::get<0>(front_instruction) = 10;
        break;
      }
      default: {
        std::cerr << "Unknown instruction type for calculation operation."
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      }
    }
  }

  if (!io_queue_.empty()) {
    auto &front_instruction =
        const_cast<decltype(io_queue_)::value_type &>(io_queue_.front());
    const auto &matrix = std::get<2>(front_instruction);
    if (!calculate_queue_.empty() && std::get<0>(front_instruction) >= 0) {
      const auto &cacl_matrix1 = std::get<2>(calculate_queue_.front());
      const auto &cacl_matrix2 = std::get<3>(calculate_queue_.front());
      if (matrix == cacl_matrix1 || matrix == cacl_matrix2) {
        // If the matrix is being used in a calculation, we cannot move it.
        return;
      }
    }
    if (std::get<0>(front_instruction) < 0 &&
        matrix->GetPosition() != Position::kReleased) {
      // In this case, we can set how much time this instruction will take.
      auto instruction_type = std::get<1>(front_instruction);
      switch (instruction_type) {
      case InstructionType::kToGpuHbm:
      case InstructionType::kToSharedMem: {
        matrix->position_ = kReleased;
        std::get<0>(front_instruction) = 300 * matrix->GetSize();
        break;
      }
      default: {
        std::cerr << "Unknown instruction type for IO operation." << std::endl;
        exit(EXIT_FAILURE);
      }
      }
    }
  }
}

bool GpuSimulator::Advance(
    bool debug_print, const MatrixMemoryAllocator *matrix_memory_allocator) {
  UpdateTimeOfInstructions();
  if (io_queue_.empty() && calculate_queue_.empty()) {
    max_gpu_hbm_size_ = std::max(max_gpu_hbm_size_, gpu_hbm_used_);
    max_shared_mem_size_ = std::max(max_shared_mem_size_, shared_mem_used_);
    return false;
  }
  bool io_not_ready = io_queue_.empty() || std::get<0>(io_queue_.front()) < 0;
  bool calc_not_ready =
      calculate_queue_.empty() || std::get<0>(calculate_queue_.front()) < 0;
  if (io_not_ready && calc_not_ready) {
    std::cerr << "Both IO and calculation queues are empty or not ready."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (io_not_ready || calc_not_ready) {
    if (io_not_ready) {
      // We only need to execute calculation instructions.
      CalculationInstruction instruction = calculate_queue_.front();
      if (std::get<0>(instruction) >= 0) {
        gpu_sim_cycle_ += std::get<0>(instruction);
        if (debug_print && matrix_memory_allocator) {
          PrintInstruction(instruction, *matrix_memory_allocator);
        }
        DoCalc(instruction);
      } else {
        std::cerr << "This CalculationInstruction will never end!" << std::endl;
        exit(EXIT_FAILURE);
      }
      calculate_queue_.pop();
      max_gpu_hbm_size_ = std::max(max_gpu_hbm_size_, gpu_hbm_used_);
      max_shared_mem_size_ = std::max(max_shared_mem_size_, shared_mem_used_);
      return true;
    } else {
      // We only need to execute IO instructions.
      IoInstruction instruction = io_queue_.front();
      if (std::get<0>(instruction) >= 0) {
        gpu_sim_cycle_ += std::get<0>(instruction);
        if (debug_print && matrix_memory_allocator) {
          PrintInstruction(instruction, *matrix_memory_allocator);
        }
        DoIO(instruction);
      } else {
        std::cerr << "This IoInstruction will never end!" << std::endl;
        exit(EXIT_FAILURE);
      }
      io_queue_.pop();
      max_gpu_hbm_size_ = std::max(max_gpu_hbm_size_, gpu_hbm_used_);
      max_shared_mem_size_ = std::max(max_shared_mem_size_, shared_mem_used_);
      return true;
    }
  }

  auto &front_io_instruction =
      const_cast<decltype(io_queue_)::value_type &>(io_queue_.front());
  auto &front_calculation_instruction =
      const_cast<decltype(calculate_queue_)::value_type &>(
          calculate_queue_.front());
  if (std::get<0>(front_io_instruction) <
      std::get<0>(front_calculation_instruction)) {
    std::get<0>(front_calculation_instruction) -=
        std::get<0>(front_io_instruction);
    gpu_sim_cycle_ += std::get<0>(front_io_instruction);
    if (debug_print && matrix_memory_allocator) {
      PrintInstruction(front_io_instruction, *matrix_memory_allocator);
    }
    DoIO(front_io_instruction);
    io_queue_.pop();
  } else {
    std::get<0>(front_io_instruction) -=
        std::get<0>(front_calculation_instruction);
    gpu_sim_cycle_ += std::get<0>(front_calculation_instruction);
    if (debug_print && matrix_memory_allocator) {
      PrintInstruction(front_calculation_instruction, *matrix_memory_allocator);
    }
    DoCalc(front_calculation_instruction);
    calculate_queue_.pop();
  }
  max_gpu_hbm_size_ = std::max(max_gpu_hbm_size_, gpu_hbm_used_);
  max_shared_mem_size_ = std::max(max_shared_mem_size_, shared_mem_used_);
  return true;
}

inline void
GpuSimulator::DoCalc(GpuSimulator::CalculationInstruction instruction) {
  auto result = std::get<5>(instruction);
  if (result == nullptr &&
      std::get<1>(instruction) != InstructionType::kRelease &&
      std::get<1>(instruction) != InstructionType::kReshape &&
      std::get<1>(instruction) != InstructionType::kTranspose) {
    std::cerr << "Result matrix is null. The type of instruction is "
              << static_cast<int>(std::get<1>(instruction)) << std::endl;
    exit(EXIT_FAILURE);
  }
  switch (std::get<1>(instruction)) {
  case InstructionType::kMatAdd: {
    *result = Matrix::MatAdd(std::get<2>(instruction), std::get<3>(instruction),
                             *this);
    break;
  };
  case InstructionType::kMatSub: {
    *result = Matrix::MatSub(std::get<2>(instruction), std::get<3>(instruction),
                             *this);
    break;
  };
  case InstructionType::kMatmul: {
    *result = Matrix::MatMul(std::get<2>(instruction), std::get<3>(instruction),
                             *this);
    break;
  }
  case InstructionType::kMatExp: {
    *result = Matrix::MatExp(std::get<2>(instruction), *this);
    break;
  }
  case InstructionType::kMatDiv: {
    *result = Matrix::MatDiv(std::get<2>(instruction), std::get<3>(instruction),
                             *this);
    break;
  }
  case InstructionType::kAddBias: {
    *result = Matrix::AddBias(std::get<2>(instruction),
                              std::get<3>(instruction), *this);
    break;
  }
  case InstructionType::kConcat: {
    *result = Matrix::Concat(std::get<2>(instruction), std::get<3>(instruction),
                             std::get<4>(instruction), *this);
    break;
  }
  case InstructionType::kGetRow: {
    *result = Matrix::GetRow(std::get<2>(instruction), std::get<4>(instruction),
                             *this);
    break;
  }
  case InstructionType::kGetColumn: {
    *result = Matrix::GetColumn(std::get<2>(instruction),
                                std::get<4>(instruction), *this);
    break;
  }
  case InstructionType::kSum: {
    *result = Matrix::Sum(std::get<2>(instruction), *this);
    break;
  }
  case InstructionType::kMulNum: {
    *result = Matrix::MulNum(std::get<2>(instruction), std::get<3>(instruction),
                             *this);
    break;
  }
  case sjtu::InstructionType::kReshape: {
    std::get<2>(instruction)
        ->Reshape(std::get<4>(instruction),
                  std::get<2>(instruction)->GetSize() /
                      std::get<4>(instruction));
    break;
  }
  case InstructionType::kRelease: {
    auto matrix = std::get<2>(instruction);
    if (matrix->GetPosition() == Position::kReleased) {
      std::cerr << "Matrix is already released." << std::endl;
      exit(EXIT_FAILURE);
    } else if (matrix->GetPosition() == Position::kInGpuHbm) {
      gpu_hbm_used_ -= matrix->GetSize();
    } else if (matrix->GetPosition() == Position::kInSharedMemory) {
      shared_mem_used_ -= matrix->GetSize();
    }
    matrix->position_ = Position::kReleased;
    break;
  }
  case InstructionType::kCopy: {
    auto src = std::get<2>(instruction);
    auto dest = std::get<5>(instruction);
    if (src->GetPosition() == Position::kReleased) {
      std::cerr << "Source matrix is released and cannot be copied."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    dest->position_ = src->GetPosition();
    dest->data_ = src->data_;
    dest->row_num_ = src->GetRowNum();
    dest->column_num_ = src->GetColumnNum();
    if (dest->GetPosition() == Position::kInGpuHbm) {
      gpu_hbm_used_ += dest->GetSize();
    } else if (dest->GetPosition() == Position::kInSharedMemory) {
      shared_mem_used_ += dest->GetSize();
    }
    break;
  }
  case InstructionType::kTranspose: {
    auto matrix = std::get<2>(instruction);
    if (matrix->GetPosition() == Position::kReleased) {
      std::cerr << "Matrix is released and cannot be transposed." << std::endl;
      exit(EXIT_FAILURE);
    }
    matrix->Transpose();
    break;
  }
  case InstructionType::kToGpuHbm:
  case InstructionType::kToSharedMem:
    std::cerr << "IO operations should not be in calculation queue."
              << std::endl; // This should not happen.
    exit(EXIT_FAILURE);
    break;
  }
}

inline void GpuSimulator::DoIO(GpuSimulator::IoInstruction instruction) {
  auto matrix = std::get<2>(instruction);
  switch (std::get<1>(instruction)) {
  case InstructionType::kToGpuHbm: {
    if (matrix->GetPosition() == Position::kInGpuHbm) {
      std::cerr << "Matrix is already in GPU HBM." << std::endl;
      return;
    }
    matrix->position_ = Position::kInGpuHbm;
    shared_mem_used_ -= matrix->GetSize();
    gpu_hbm_used_ += matrix->GetSize();
    break;
  }
  case InstructionType::kToSharedMem: {
    if (matrix->GetPosition() == Position::kInSharedMemory) {
      std::cerr << "Matrix is already in shared memory." << std::endl;
      return;
    }
    matrix->position_ = Position::kInSharedMemory;
    gpu_hbm_used_ -= matrix->GetSize();
    shared_mem_used_ += matrix->GetSize();
    break;
  }
  case InstructionType::kTranspose:
  case InstructionType::kRelease:
  case InstructionType::kReshape:
  case InstructionType::kMatAdd:
  case InstructionType::kMatSub:
  case InstructionType::kMatmul:
  case InstructionType::kMatExp:
  case InstructionType::kMatDiv:
  case InstructionType::kAddBias:
  case InstructionType::kMulNum:
  case InstructionType::kConcat:
  case InstructionType::kGetRow:
  case InstructionType::kGetColumn:
  case InstructionType::kSum:
  case InstructionType::kCopy: {
    std::cerr << "Calculation operations should not be in IO queue."
              << std::endl; // This should not happen.
    exit(EXIT_FAILURE);
    break;
  }
  }
}

Matrix *Rater::GetNextQuery() {
  if (!next_query_available_ ||
      current_query_index_ >= static_cast<int>(queries_.size())) {
    std::cerr << "Next query is not available." << std::endl;
    exit(EXIT_FAILURE);
  }
  next_query_available_ = false;
  return queries_[current_query_index_];
}

bool isEqual(float a, float b, float absoluteEpsilon = 1e-6f,
             float relativeEpsilon = 1e-5f) {
  if (a == b)
    return true;
  float diff = std::abs(a - b);
  if (diff < absoluteEpsilon)
    return true;
  float largest = std::max(std::abs(a), std::abs(b));
  return diff <= relativeEpsilon * largest;
}

void Rater::CommitAnswer(Matrix &answer) {
  if (next_query_available_) {
    std::cerr << "You must call GetNextQuery() before committing an answer.";
    exit(EXIT_FAILURE);
  }
  assert(current_query_index_ < static_cast<int>(answers_.size()));
  if (answer.GetPosition() != Position::kInGpuHbm) {
    errors_.push_back(answers_[current_query_index_]->GetSize());
    std::cerr << "Answer matrix must be in GPU HBM." << std::endl;
  } else if (answer.GetColumnNum() !=
                 answers_[current_query_index_]->GetColumnNum() ||
             answer.GetRowNum() !=
                 answers_[current_query_index_]->GetRowNum()) {
    errors_.push_back(answers_[current_query_index_]->GetSize());
    std::cerr << "Answer matrix size mismatch." << std::endl;
  } else {
    int error_count = 0;
    for (size_t i = 0; i < answer.GetSize(); ++i) {
      if (!isEqual(answer.data_[i], answers_[current_query_index_]->data_[i])) {
        error_count++;
      }
    }
    errors_.push_back(error_count);
  }
  answer.position_ = Position::kReleased;
  next_query_available_ = true;
  current_query_index_++;
}

void Rater::PrintResult(const GpuSimulator &gpu_sim) const {
  std::cerr << "Error Rate: ";
  float total_ans = 0;
  for (const auto &ans : answers_) {
    total_ans += ans->GetSize();
  }
  std::cerr << std::fixed << std::setprecision(6)
            << GetErrorsCount() / total_ans << std::endl;
  std::cerr << "Max GPU HBM used: " << gpu_sim.max_gpu_hbm_size_ << std::endl;
  std::cerr << "Max Shared Memory used: " << gpu_sim.max_shared_mem_size_
            << std::endl;
  std::cerr << "Total GPU cycles: " << gpu_sim.gpu_sim_cycle_ << std::endl;
}

void GpuSimulator::PrintInstruction(
    const CalculationInstruction &instruction,
    const MatrixMemoryAllocator &matrix_memory_allocator) const {
  auto instruction_type = std::get<1>(instruction);
  auto matrix1 = std::get<2>(instruction);
  auto matrix2 = std::get<3>(instruction);
  auto result = std::get<5>(instruction);
  switch (instruction_type) {
  case InstructionType::kMatAdd: {
    std::cerr << "MatAdd: ";
    break;
  }
  case InstructionType::kMatSub: {
    std::cerr << "MatSub: ";
    break;
  }
  case InstructionType::kMatmul: {
    std::cerr << "MatMul: ";
    break;
  }
  case InstructionType::kMatExp: {
    std::cerr << "MatExp: ";
    break;
  }
  case InstructionType::kMatDiv: {
    std::cerr << "MatDiv: ";
    break;
  }
  case InstructionType::kAddBias: {
    std::cerr << "AddBias: ";
    break;
  }
  case InstructionType::kConcat: {
    std::cerr << "Concat: ";
    break;
  }
  case InstructionType::kGetRow: {
    std::cerr << "GetRow: ";
    break;
  }
  case InstructionType::kGetColumn: {
    std::cerr << "GetColumn: ";
    break;
  }
  case InstructionType::kSum: {
    std::cerr << "Sum: ";
    break;
  }
  case InstructionType::kMulNum: {
    std::cerr << "MulNum: ";
    break;
  }
  case InstructionType::kReshape: {
    std::cerr << "Reshape: ";
    break;
  }
  case InstructionType::kTranspose: {
    std::cerr << "Transpose: ";
    break;
  }
  case InstructionType::kCopy: {
    std::cerr << "Copy: ";
    break;
  }
  case InstructionType::kRelease: {
    std::cerr << "Release: ";
    break;
  }
  case InstructionType::kToGpuHbm:
  case InstructionType::kToSharedMem: {
    std::cerr << "IO operation should not be in calculation queue! please "
                 "contact TA.";
    exit(EXIT_FAILURE); // These should not be in calculation queue.
  }
  }
  if (matrix1 != nullptr) {
    std::cerr << "Matrix1: " << matrix_memory_allocator.GetMatrixName(matrix1);
  }
  if (matrix2 != nullptr) {
    std::cerr << ", Matrix2: "
              << matrix_memory_allocator.GetMatrixName(matrix2);
  }
  if (result != nullptr) {
    std::cerr << ", Result: " << matrix_memory_allocator.GetMatrixName(result);
  }
  std::cerr << std::endl;
}

void GpuSimulator::PrintInstruction(
    const IoInstruction &instruction,
    const MatrixMemoryAllocator &matrix_memory_allocator) const {
  auto instruction_type = std::get<1>(instruction);
  auto matrix = std::get<2>(instruction);
  switch (instruction_type) {
  case InstructionType::kToGpuHbm: {
    std::cerr << "Move to GPU HBM: ";
    break;
  }
  case InstructionType::kToSharedMem: {
    std::cerr << "Move to Shared Memory: ";
    break;
  }
  default: {
    std::cerr << "Calculation operation should not be in IO queue! please "
                 "contact TA.";
    exit(EXIT_FAILURE);
  }
  }
  if (matrix != nullptr) {
    std::cerr << "Matrix: " << matrix_memory_allocator.GetMatrixName(matrix);
  }
  std::cerr << std::endl;
}

} // namespace sjtu

#endif // SIMULATION_HPP