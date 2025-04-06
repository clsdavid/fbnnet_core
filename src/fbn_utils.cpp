#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>

namespace py = pybind11;

// ----- Utility Functions -----

std::string to_string(double val) {
    std::ostringstream stm;
    stm << val;
    return stm.str();
}

std::string mpaste(const std::vector<std::string>& x, const std::string& sep) {
    std::string res;
    for (size_t i = 0; i < x.size(); ++i) {
        res += x[i];
        if (i != x.size() - 1) res += sep;
    }
    return res;
}

template <typename T>
std::vector<T> concatenator(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// ----- Matrix Binding (NumPy) -----

py::array_t<double> mcbind(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();
    if (a_buf.ndim != 2 || b_buf.ndim != 2)
        throw std::runtime_error("Inputs must be 2D matrices");
    if (a_buf.shape[0] != b_buf.shape[0])
        throw std::runtime_error("Row count mismatch");

    size_t rows = a_buf.shape[0];
    size_t cols = a_buf.shape[1] + b_buf.shape[1];
    py::array_t<double> out({rows, cols});
    py::buffer_info out_buf = out.request();
    
    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    
    // Copy data column-wise
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < a_buf.shape[1]; ++j)
            out_ptr[i * cols + j] = a_ptr[i * a_buf.shape[1] + j];
        for (size_t j = 0; j < b_buf.shape[1]; ++j)
            out_ptr[i * cols + a_buf.shape[1] + j] = b_ptr[i * b_buf.shape[1] + j];
    }
    return out;
}

// ----- Other Core Functions -----

bool isReallyNA(double val) {
    return std::isnan(val);
}

int countZeros(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int count = 0;
    for (ssize_t i = 0; i < buf.size; ++i)
        if (ptr[i] == 0) ++count;
    return count;
}

// ... (Additional converted functions follow similar patterns)

// ----- PyBind11 Module Definition -----
PYBIND11_MODULE(fbn_utils, m) {
    m.def("to_string", &to_string);
    m.def("mpaste", &mpaste);
    m.def("concatenator", &concatenator<std::string>);
    m.def("concatenatorI", &concatenator<int>);
    m.def("concatenatorN", &concatenator<double>);
    m.def("mcbind", &mcbind);
    m.def("isReallyNA", &isReallyNA);
    m.def("countZeros", &countZeros);
    // ... Bind other functions
}