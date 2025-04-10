#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "fbn_matrix.h"

namespace py = pybind11;

py::array_t<double> FBNMatrix::matrix_to_numpy() const {
    if (matrix_.empty() || matrix_[0].empty()) {
        return py::array_t<double>(py::array::ShapeContainer{0, 0});
    }

    size_t rows = matrix_.size();
    size_t cols = matrix_[0].size();

    py::array_t<double> array(py::array::ShapeContainer{rows, cols});
    auto buffer = array.mutable_unchecked<2>();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            buffer(i, j) = matrix_[i][j];
        }
    }

    return array;
}


void FBNMatrix::validate_dimensions() const {
    if (!row_names_.empty() && row_names_.size() != matrix_.size()) {
        throw std::runtime_error("Row names size doesn't match matrix rows");
    }
    if (!col_names_.empty() && !matrix_.empty() && col_names_.size() != matrix_[0].size()) {
        throw std::runtime_error("Column names size doesn't match matrix columns");
    }
}

FBNMatrix::FBNMatrix(const std::vector<std::vector<double>>& matrix,
                                 const std::vector<std::string>& row_names,
                                 const std::vector<std::string>& col_names)
    : matrix_(matrix), row_names_(row_names), col_names_(col_names) {
    matrix_t_ = matrix_to_numpy();    
    validate_dimensions();
}

FBNMatrix::FBNMatrix(py::array_t<double> mat,
                                 const std::vector<std::string>& row_names,
                                 const std::vector<std::string>& col_names) {
    auto buf = mat.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Matrix must be 2-dimensional");
    }

    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    matrix_.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        matrix_[i].resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            matrix_[i][j] = ptr[i * cols + j];
        }
    }

    row_names_ = row_names;
    col_names_ = col_names;
    matrix_t_ = matrix_to_numpy();
    validate_dimensions();
}

void FBNMatrix::print(size_t precision) const {
    std::cout << std::fixed << std::setprecision(precision);
    
    if (!col_names_.empty()) {
        std::cout << std::setw(12) << " ";
        for (const auto& name : col_names_) {
            std::cout << std::setw(12) << name;
        }
        std::cout << "\n";
    }
    
    for (size_t i = 0; i < matrix_.size(); ++i) {
        if (i < row_names_.size()) {
            std::cout << std::setw(12) << row_names_[i];
        } else {
            std::cout << std::setw(12) << "Row " << i+1;
        }
        
        for (const auto& val : matrix_[i]) {
            std::cout << std::setw(12) << val;
        }
        std::cout << "\n";
    }
}

std::string FBNMatrix::to_string() const {
    std::ostringstream oss;
    if (!col_names_.empty()) {
        oss << "Col names: ";
        for (const auto& name : col_names_) {
            oss << name << " ";
        }
        oss << "\n";
    }
    
    for (size_t i = 0; i < matrix_.size(); ++i) {
        if (i < row_names_.size()) {
            oss << row_names_[i] << ": ";
        }
        
        for (const auto& val : matrix_[i]) {
            oss << val << " ";
        }
        oss << "\n";
    }
    return oss.str();
}