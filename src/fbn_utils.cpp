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
    /*
        * Concatenate a vector of strings with a separator.
        * If the vector is empty, return an empty string.
        * If the separator is empty, return the first element of the vector.
    */
    std::string res;
    for (size_t i = 0; i < x.size(); ++i) {
        res += x[i];
        if (i != x.size() - 1) res += sep;
    }
    return res;
}

template <typename T>
std::vector<T> concatenate(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}


double dround(double val, int decimal) {
    double factor = std::pow(10.0, decimal);
    return std::round(val * factor) / factor;
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

py::array_t<double> mrbind(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info a_buf = a.request(), b_buf = b.request();
    if (a_buf.ndim != 2 || b_buf.ndim != 2)
        throw std::runtime_error("Inputs must be 2D matrices");
    if (a_buf.shape[0] != b_buf.shape[0])
        throw std::runtime_error("Row count mismatch");

    size_t rows = a_buf.shape[0] + b_buf.shape[0];
    size_t cols = a_buf.shape[1];
    py::array_t<double> out({rows, cols});
    py::buffer_info out_buf = out.request();
    
    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    
    // Copy data row-wise
    for (size_t i = 0; i < a_buf.shape[0]; ++i) {
        for (size_t j = 0; j < a_buf.shape[1]; ++j)
            out_ptr[i * cols + j] = a_ptr[i * a_buf.shape[1] + j];
    }
    for (size_t i = 0; i < b_buf.shape[0]; ++i) {
        for (size_t j = 0; j < b_buf.shape[1]; ++j)
            out_ptr[(i + a_buf.shape[0]) * cols + j] = b_ptr[i * b_buf.shape[1] + j];
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
std::vector<T> vector_sort(std::vector<T> x, bool dsc = true) {
    if (dsc) {
        std::sort(x.rbegin(), x.rend());
    } else {
        std::sort(x.begin(), x.end());
    }
    return x;
}
std::vector<std::string> convertStringIntoVector(std::string value, int outputType = 1, bool lowerCase = false) {
    std::vector<std::string> result;
    std::istringstream iss(value);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (lowerCase) {
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        }
        result.push_back(token);
    }
    return result;
}
std::vector<bool> a_in_b(const std::vector<std::string>& names1, const std::vector<std::string>& names2) {
    std::vector<bool> result(names1.size(), false);
    for (size_t i = 0; i < names1.size(); ++i) {
        result[i] = std::find(names2.begin(), names2.end(), names1[i]) != names2.end();
    }
    return result;
}
std::vector<int> a_in_b_index(const std::vector<std::string>& names1, const std::vector<std::string>& names2) {
    std::vector<int> result;
    for (size_t i = 0; i < names1.size(); ++i) {
        auto it = std::find(names2.begin(), names2.end(), names1[i]);
        if (it != names2.end()) {
            result.push_back(std::distance(names2.begin(), it));
        }
    }
    return result;
}
py::list resizel(const py::list& x, int n) {
    py::list result(n);
    for (int i = 0; i < n; ++i) {
        if (i < x.size()) {
            result[i] = x[i];
        } else {
            result[i] = py::none();
        }
    }
    return result;
}
py::list orderByname(const py::list& x, const std::vector<std::string>& names) {
    py::list result;
    for (const auto& name : names) {
        for (const auto& item : x) {
            if (item.cast<std::string>() == name) {
                result.append(item);
                break;
            }
        }
    }
    return result;
}
py::list removeEmptyElement(const py::list& x) {
    py::list result;
    for (const auto& item : x) {
        if (!item.is_none()) {
            result.append(item);
        }
    }
    return result;
}
py::array_t<double> substractM(py::array_t<double> m, py::array_t<double> v) {
    auto m_buf = m.request(), v_buf = v.request();
    if (m_buf.ndim != 2 || v_buf.ndim != 1)
        throw std::runtime_error("Matrix and vector dimensions do not match");
    if (m_buf.shape[1] != v_buf.shape[0])
        throw std::runtime_error("Column count mismatch");

    size_t rows = m_buf.shape[0];
    size_t cols = m_buf.shape[1];
    py::array_t<double> out({rows, cols});
    auto out_buf = out.mutable_unchecked<2>();
    
    double* m_ptr = static_cast<double*>(m_buf.ptr);
    double* v_ptr = static_cast<double*>(v_buf.ptr);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out_buf(i, j) = m_ptr[i * cols + j] - v_ptr[j];
        }
    }
    return out;
}
int matchCount(py::array_t<double> m, py::array_t<double> v) {
    auto m_buf = m.request(), v_buf = v.request();
    if (m_buf.ndim != 2 || v_buf.ndim != 1)
        throw std::runtime_error("Matrix and vector dimensions do not match");
    if (m_buf.shape[1] != v_buf.shape[0])
        throw std::runtime_error("Column count mismatch");

    int count = 0;
    double* m_ptr = static_cast<double*>(m_buf.ptr);
    double* v_ptr = static_cast<double*>(v_buf.ptr);
    
    for (size_t i = 0; i < m_buf.shape[0]; ++i) {
        for (size_t j = 0; j < m_buf.shape[1]; ++j) {
            if (m_ptr[i * m_buf.shape[1] + j] == v_ptr[j]) {
                ++count;
            }
        }
    }
    return count;
}

// ----- PyBind11 Module Definition -----
PYBIND11_MODULE(fbnnet_core, m) {
    m.def("to_string", &to_string);
    m.def("mpaste", &mpaste);
    m.def("dround", &dround);
    m.def("concatenate", &concatenate<std::string>);
    m.def("concatenateI", &concatenate<int>);
    m.def("concatenateN", &concatenate<double>);
    m.def("mcbind", &mcbind);
    m.def("mrbind", &mrbind);
    m.def("isReallyNA", &isReallyNA);
    m.def("countZeros", &countZeros);
    // ... Bind other functions
}