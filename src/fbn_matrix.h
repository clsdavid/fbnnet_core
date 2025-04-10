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