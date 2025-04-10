#ifndef FBN_Matrix_H
#define FBN_Matrix_H

#include <vector>
#include <string>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class FBNMatrix {
private:
    std::vector<std::vector<double>> matrix_;
    std::vector<std::string> row_names_;
    std::vector<std::string> col_names_;
    py::array_t<double> matrix_t_;   //numpy array representation
    void validate_dimensions() const;

public:
    // Constructors
    FBNMatrix() = default;
    
    // From vector<vector<double>>
    FBNMatrix(const std::vector<std::vector<double>>& matrix,
                    const std::vector<std::string>& row_names = {},
                    const std::vector<std::string>& col_names = {});
    
    // From numpy array
    FBNMatrix(py::array_t<double> mat,
                    const std::vector<std::string>& row_names = {},
                    const std::vector<std::string>& col_names = {});

    // Accessors
    const std::vector<std::vector<double>>& matrix() const { return matrix_; }
    const py::array_t<double>& matrix_t() const {
        if (matrix_t_.size() == 0) {
            matrix_t_ = py::array_t<double>(matrix_.size(), matrix_[0].size(), matrix_[0].data());
        }
        return matrix_t_;
    }
    const std::vector<double>& operator[](size_t index) const {
        if (index >= matrix_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return matrix_[index];
    }
    const std::vector<std::string>& row_names() const { return row_names_; }
    const std::vector<std::string>& col_names() const { return col_names_; }

    // Dimensions
    size_t num_rows() const { return matrix_.size(); }
    size_t num_cols() const { return matrix_.empty() ? 0 : matrix_[0].size(); }

    // Output
    void print(size_t precision = 4) const;
    std::string to_string() const;
};

#endif // FBN_Matrix_H