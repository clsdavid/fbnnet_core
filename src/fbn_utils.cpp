#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include "fbn_utils.h"

namespace py = pybind11;

// ----- Utility Functions -----

void debug_function(const py::object& obj) {
    // Print Python-style representation
    py::print("[DEBUG from C++]", py::str(obj));
}

void debug_str(const std::string& msg) {
    // Option 1: Print via std::cout
    std::cout << "[DEBUG] " << msg << std::endl;
}

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

// Helper function to concatenate row names
std::vector<std::string> concatenate_row_names(
    const std::vector<std::string>& a_names,
    const std::vector<std::string>& b_names) {
    std::vector<std::string> result;
    result.reserve(a_names.size() + b_names.size());
    result.insert(result.end(), a_names.begin(), a_names.end());
    result.insert(result.end(), b_names.begin(), b_names.end());
    return result;
}

std::vector<std::string> concatenate_col_names(
    const std::vector<std::string>& a_names,
    const std::vector<std::string>& b_names) {
    std::vector<std::string> result;
    result.reserve(a_names.size() + b_names.size());
    result.insert(result.end(), a_names.begin(), a_names.end());
    result.insert(result.end(), b_names.begin(), b_names.end());
    return result;
}

// Concatenate two numeric matrices by columns
py::array_t<double> mcbind(py::array_t<double> a, py::array_t<double> b) {
    // Get matrix dimensions
    auto a_buf = a.request();
    auto b_buf = b.request();
    
    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("Both inputs must be 2D matrices");
    }
    
    size_t a_rows = a_buf.shape[0];
    size_t a_cols = a_buf.shape[1];
    size_t b_rows = b_buf.shape[0];
    size_t b_cols = b_buf.shape[1];
    
    if (a_rows != b_rows) {
        std::string msg = "The two matrices must have the same number of rows: nrow(a)=";
        msg += std::to_string(a_rows);
        msg += ", nrow(b)=";
        msg += std::to_string(b_rows);
        throw std::runtime_error(msg);
    }
    
    // Create output matrix
    size_t out_rows = a_rows;
    size_t out_cols = a_cols + b_cols;
    auto out = py::array_t<double>({out_rows, out_cols});
    auto out_buf = out.request();
    
    // Get pointers to data
    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    
    // Copy data (column-wise concatenation)
    for (size_t i = 0; i < out_rows; i++) {
        // Copy columns from matrix a
        for (size_t j = 0; j < a_cols; j++) {
            out_ptr[i * out_cols + j] = a_ptr[i * a_cols + j];
        }
        // Copy columns from matrix b
        for (size_t j = 0; j < b_cols; j++) {
            out_ptr[i * out_cols + a_cols + j] = b_ptr[i * b_cols + j];
        }
    }
    
    // Handle row names if they exist (copy from first matrix)
    if (py::hasattr(a, "row_names")) {
        out.attr("row_names") = a.attr("row_names");
    }
    
    // Handle column names if they exist in both matrices
    if (py::hasattr(a, "col_names") && py::hasattr(b, "col_names")) {
        auto a_col_names = a.attr("col_names").cast<std::vector<std::string>>();
        auto b_col_names = b.attr("col_names").cast<std::vector<std::string>>();
        auto combined_col_names = concatenate_col_names(a_col_names, b_col_names);
        out.attr("col_names") = py::cast(combined_col_names);
    }
    
    return out;
}

// Concatenate two numeric matrices by rows
py::array_t<double> mrbind(py::array_t<double> a, py::array_t<double> b) {
    // Get matrix dimensions
    auto a_buf = a.request();
    auto b_buf = b.request();
    
    if (a_buf.ndim != 2 || b_buf.ndim != 2) {
        throw std::runtime_error("Both inputs must be 2D matrices");
    }
    
    size_t a_rows = a_buf.shape[0];
    size_t a_cols = a_buf.shape[1];
    size_t b_rows = b_buf.shape[0];
    size_t b_cols = b_buf.shape[1];
    
    if (a_cols != b_cols) {
        std::string msg = "The two matrices must have the same number of columns: ncol(a)=";
        msg += std::to_string(a_cols);
        msg += ", ncol(b)=";
        msg += std::to_string(b_cols);
        throw std::runtime_error(msg);
    }
    
    // Create output matrix
    size_t out_rows = a_rows + b_rows;
    size_t out_cols = a_cols;
    auto out = py::array_t<double>({out_rows, out_cols});
    auto out_buf = out.request();
    
    // Get pointers to data
    double* a_ptr = static_cast<double*>(a_buf.ptr);
    double* b_ptr = static_cast<double*>(b_buf.ptr);
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    
    // Copy data
    for (size_t i = 0; i < a_rows; i++) {
        for (size_t j = 0; j < a_cols; j++) {
            out_ptr[i * out_cols + j] = a_ptr[i * a_cols + j];
        }
    }
    
    for (size_t i = 0; i < b_rows; i++) {
        for (size_t j = 0; j < b_cols; j++) {
            out_ptr[(a_rows + i) * out_cols + j] = b_ptr[i * b_cols + j];
        }
    }
    
    // Handle row names if they exist (as Python lists in the array's .row_names attribute)
    if (py::hasattr(a, "row_names") && py::hasattr(b, "row_names")) {
        auto a_row_names = a.attr("row_names").cast<std::vector<std::string>>();
        auto b_row_names = b.attr("row_names").cast<std::vector<std::string>>();
        auto combined_row_names = concatenate_row_names(a_row_names, b_row_names);
        out.attr("row_names") = py::cast(combined_row_names);
    }
    
    // Handle column names if they exist (copy from first matrix)
    if (py::hasattr(a, "col_names")) {
        out.attr("col_names") = a.attr("col_names");
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
template <typename T>
std::vector<T> vector_sort(std::vector<T> x, bool dsc) {
    if (dsc) {
        std::sort(x.rbegin(), x.rend());
    } else {
        std::sort(x.begin(), x.end());
    }
    return x;
}
std::vector<std::string> convertStringIntoVector(std::string value, int outputType, bool lowerCase) {
    int len = value.length();
    std::vector<std::string> res;
    std::string t;
    
    for(int i = 0; i < len; i++) {
        char v = value[i];
        if(lowerCase) {
            v = std::tolower(v);
        }
        if(v == ' ') {
            if(t.length() > 0) {
                res.push_back(t);
                t.clear();
            }
            continue;
        }

        if(outputType == 1 || outputType == 0) {
            if(v == '&' || v == ',' || v == '!') {
                if(t.length() > 0) {
                    res.push_back(t);
                    t.clear();
                }

                if(outputType == 1) {
                    t.push_back(v);
                } else {
                    if(v != '!') {
                        t.push_back(v);
                    }
                }
                res.push_back(t);
                t.clear();
                continue;
            }

            if(outputType == 1) {
                t.push_back(v);
            } else {
                if(v != '!') {
                    t.push_back(v);
                }
            }
        } else {
            if(v == '&' || v == ',') {
                if(t.length() > 0) {
                    res.push_back(t);
                    t.clear();
                }
                t.push_back(v);
                res.push_back(t);
                t.clear();
                continue;
            }
            t.push_back(v);
        }
    }
    
    if(t.length() > 0) {
        res.push_back(t);
    }

    return res;
}
// Returns a boolean mask indicating which elements of names1 are in names2
std::vector<bool> a_in_b(const std::vector<std::string>& names1, 
    const std::vector<std::string>& names2) {
    std::vector<bool> res(names1.size(), false);

    if (names2.empty()) {
        return std::vector<bool>(names1.size(), false);
    }
    if (names1.empty()) {
        return res;
    }

    std::unordered_set<std::string> names2_set(names2.begin(), names2.end());

    for (size_t i = 0; i < names1.size(); ++i) {
        res[i] = (names2_set.find(names1[i]) != names2_set.end());
    }

    return res;
}

// Returns indices of elements in names2 that are in names1
std::vector<size_t> a_in_b_index(const std::vector<std::string>& names1,
            const std::vector<std::string>& names2) {
    std::vector<size_t> res;

    if (names1.empty() || names2.empty()) {
        return res;
    }

    std::unordered_set<std::string> names1_set(names1.begin(), names1.end());

    for (size_t i = 0; i < names2.size(); ++i) {
        if (names1_set.find(names2[i]) != names1_set.end()) {
            res.push_back(i);
        }
    }

    return res;
}

// Returns a boolean mask indicating which elements of names1 are not in names2
std::vector<bool> a_not_in_b(const std::vector<std::string>& names1, const std::vector<std::string>& names2) {
    std::vector<bool> res(names1.size(), false);

    if (names2.empty()) {
        return std::vector<bool>(names1.size(), true);
    }
    if (names1.empty()) {
        return res;
    }

    std::unordered_set<std::string> names2_set(names2.begin(), names2.end());

    for (size_t i = 0; i < names1.size(); ++i) {
        res[i] = (names2_set.find(names1[i]) == names2_set.end());
    }

    return res;
}

// Returns indices of elements in names1 that are not in names2
std::vector<size_t> a_not_in_b_index(const std::vector<std::string>& names1, const std::vector<std::string>& names2) {
    std::vector<size_t> res;

    if (names1.empty()) {
        return res;
    }
    if (names2.empty()) {
        res.resize(names1.size());
        std::iota(res.begin(), res.end(), 0);
        return res;
    }

    std::unordered_set<std::string> names2_set(names2.begin(), names2.end());

    for (size_t i = 0; i < names1.size(); ++i) {
        if (names2_set.find(names1[i]) == names2_set.end()) {
            res.push_back(i);
        }
    }

    return res;
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
py::dict orderByName(const py::dict& x, const std::vector<std::string>& names) {
    py::dict y;
    
    // Reorder the dictionary based on the names vector
    for (const auto& name : names) {
        if (x.contains(name)) {
            y[name.c_str()] = x[name.c_str()];
        } else {
            throw std::runtime_error("Key '" + name + "' not found in input dictionary");
        }
    }
    
    return y;
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

py::array_t<double> subtractM(py::array_t<double>& m, py::array_t<double>& v) {
    // Access the input arrays
    auto m_buf = m.request();
    auto v_buf = v.request();
    
    // Get dimensions
    size_t nrow = m_buf.shape[0];
    size_t ncol = m_buf.shape[1];
    
    // Check vector length matches matrix rows
    if (v_buf.size != nrow) {
        throw std::runtime_error("Vector length must match number of matrix rows");
    }
    
    // Create output array (row-major)
    py::array_t<double> res({nrow, ncol});
    auto res_buf = res.request();
    
    // Get pointers to the data
    double* m_ptr = static_cast<double*>(m_buf.ptr);
    double* v_ptr = static_cast<double*>(v_buf.ptr);
    double* res_ptr = static_cast<double*>(res_buf.ptr);
    
    // Perform subtraction (row-major traversal)
    for (size_t row = 0; row < nrow; ++row) {
        for (size_t col = 0; col < ncol; ++col) {
            res_ptr[row * ncol + col] = std::abs(m_ptr[row * ncol + col] - v_ptr[row]);
        }
    }
    
    return res;
}

int matchCount(py::array_t<double>& m, py::array_t<double>& v) {
    py::array_t<double> diff = subtractM(m, v);
    auto diff_buf = diff.request();
    size_t nrow = diff_buf.shape[0];
    size_t ncol = diff_buf.shape[1];
    double* diff_ptr = static_cast<double*>(diff_buf.ptr);

    std::vector<double> col_sums(ncol, 0.0);

    // Correct row-major traversal (row first, then column)
    for (size_t row = 0; row < nrow; ++row) {
        for (size_t col = 0; col < ncol; ++col) {
            col_sums[col] += diff_ptr[row * ncol + col];  // Row-major indexing
        }
    }

    return std::count(col_sums.begin(), col_sums.end(), 0.0);
}

// add back the missing functions
py::dict fisher_test_cpp(py::array_t<double>& x, double conf_level) {
    try {
        // This for 2x2 table
        // Import statsmodels
        py::module statsmodels = py::module::import("statsmodels.stats.contingency_tables");
        
        // Reshape input to 2x2 table (SciPy/statsmodels expect this format)
        py::array_t<double> table = x.attr("reshape")(std::make_tuple(2, 2));
        
        // Create a 2x2 contingency table object
        py::object table_obj = statsmodels.attr("Table2x2")(table);
        
        // Perform Fisher's exact test
        py::object result = table_obj.attr("test_nominal_association")();
        
        // Extract results (similar to R's fisher.test())
        py::dict test_out;
        test_out["p_value"] = result.attr("pvalue");
        test_out["estimate"] = table_obj.attr("oddsratio");
        test_out["conf_int"] = table_obj.attr("oddsratio_confint")(conf_level);
        
        return test_out;
    } 
    catch (const std::exception &e) {
        // Handle errors (e.g., statsmodels not installed)
        throw std::runtime_error("Error in fisher_test_cpp: " + std::string(e.what()));
    }
}

std::vector<std::string> subCPP(
    const std::vector<std::string>& pattern,
    const std::vector<std::string>& replacement,
    const std::vector<std::string>& x) {
    std::vector<std::string> y = x;  // Initialize output with copy of input
    size_t patlen = pattern.size();
    size_t replen = replacement.size();

    if (patlen != replen) {
        py::print("Error: Pattern and replacement length do not match");
        return y;
    }

    for (size_t i = 0; i < patlen; ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            if (x[j] == pattern[i]) {
                y[j] = replacement[i];
            }
        }
    }
    return y;
}

// String vector sorting
std::vector<std::string> char_sort(std::vector<std::string> x, bool dsc) {
    if (dsc) {
        std::sort(x.begin(), x.end(), std::greater<std::string>());
    } else {
        std::sort(x.begin(), x.end());
    }
    return x;
}

// Integer vector sorting
std::vector<size_t> int_sort(std::vector<size_t> x, bool dsc) {
    if (dsc) {
        std::sort(x.begin(), x.end(), std::greater<size_t>());
    } else {
        std::sort(x.begin(), x.end());
    }
    return x;
}

// Double vector sorting
std::vector<double> num_sort(std::vector<double> x, bool dsc) {
    if (dsc) {
        std::sort(x.begin(), x.end(), std::greater<double>());
    } else {
        std::sort(x.begin(), x.end());
    }
    return x;
}

std::vector<std::string> splitExpression(const std::string& expression,
    int outputType,
    bool lowerCase) {
    std::vector<std::string> res;

    if (expression == "1" || expression == "0") {
        res.push_back(expression);
    } else {
        if (outputType == 1) {
            res = convertStringIntoVector(expression, 1, lowerCase);
        } else {
            res = convertStringIntoVector(expression, 2, lowerCase);
        }
    }
    return res;
}

std::string join_vector(const std::vector<std::string>& vec, const std::string& sep) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) oss << sep;
    }
    return std::string(oss.str());
}

std::string join_double(const std::vector<double>& vec, const std::string& sep) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) oss << sep;
    }
    return std::string(oss.str());
}

//create a function that deep copy py::dict
py::dict deep_copy_dict(const py::dict& x) {
    py::dict y;
    for (const auto& item : x) {
        y[item.first] = item.second;
    }
    return y;
}

// ----- PyBind11 Module Definition -----
PYBIND11_MODULE(fbnnet_utils, m) {
    m.def("debug_function", &debug_function, "Print debug message from C++");
    m.def("debug_str", &debug_str, "Print debug message from C++");
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
    m.def("vector_sort", &vector_sort<std::string>);
    m.def("convertStringIntoVector", &convertStringIntoVector);
    m.def("a_in_b", &a_in_b);
    m.def("a_in_b_index", &a_in_b_index);
    m.def("resizel", &resizel);
    m.def("orderByName", &orderByName, 
        "Reorder dictionary items according to specified names",
        py::arg("x"), py::arg("names"));
    m.def("removeEmptyElement", &removeEmptyElement);
    m.def("subtractM", &subtractM, "Subtract vector from matrix columns and take absolute value");
    m.def("matchCount", &matchCount, "Count how many matrix columns exactly match the vector");

    m.def("fisher_test_cpp", &fisher_test_cpp,
        "Perform Fisher's exact test (like R's fisher.test)",
        py::arg("x"), py::arg("conf_level") = 0.95);
    m.def("sub_cpp", &subCPP, "A function that substitutes patterns in strings",
        py::arg("pattern"), py::arg("replacement"), py::arg("x"));

    m.def("char_sort", &char_sort, "Sort a vector of strings",
            py::arg("x"), py::arg("dsc") = false);
    m.def("int_sort", &int_sort, "Sort a vector of integers",
            py::arg("x"), py::arg("dsc") = false);
    m.def("num_sort", &num_sort, "Sort a vector of doubles",
            py::arg("x"), py::arg("dsc") = false);
    m.def("a_not_in_b", &a_not_in_b, 
            "Returns boolean mask of elements in first array not in second array",
            py::arg("names1"), py::arg("names2"));
          
    m.def("a_not_in_b_index", &a_not_in_b_index,
            "Returns indices of elements in first array not in second array",
            py::arg("names1"), py::arg("names2"));

    m.def("splitExpression", &splitExpression,
        "Split an expression into a vector of inputs",
        py::arg("expression"),
        py::arg("output_type"),
        py::arg("lower_case") = false);
    m.def("join_vector", &join_vector,
        "Join a vector of strings into a single string",
        py::arg("vec"), py::arg("sep") = ", ");
    m.def("join_double", &join_double,
        "Join a vector of doubles into a single string",
        py::arg("vec"), py::arg("sep") = ", ");
    m.def("deep_copy_dict", &deep_copy_dict,
        "Deep copy a Python dictionary",
        py::arg("x"));
    // ... Bind other functions
}