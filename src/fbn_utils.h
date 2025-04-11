#ifndef FBN_UTILS_H
#define FBN_UTILS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// ----- Function Declarations -----
void debug_function(const py::object& obj);
void debug_str(const std::string& msg);
std::string to_string(double val);
std::string mpaste(const std::vector<std::string>& x, const std::string& sep = "");

// Template for vector concatenation
template <typename T>
std::vector<T> concatenate(const std::vector<T>& a, const std::vector<T>& b);


// Matrix operations
std::vector<std::string> concatenate_row_names(
    const std::vector<std::string>& a_names,
    const std::vector<std::string>& b_names);
std::vector<std::string> concatenate_col_names(
    const std::vector<std::string>& a_names,
    const std::vector<std::string>& b_names);
py::array_t<double> mcbind(py::array_t<double> a, py::array_t<double> b);
py::array_t<double> mrbind(py::array_t<double> a, py::array_t<double> b);

// Numeric utilities
bool isReallyNA(double val);
int countZeros(py::array_t<double> arr);
double dround(double val, int decimal);

// Vector operations
template <typename T>
std::vector<T> vector_sort(std::vector<T> x, bool dsc = true);

// String operations
std::vector<std::string> convertStringIntoVector(std::string value, 
                                               int outputType = 1, 
                                               bool lowerCase = false);

// Set operations
std::vector<bool> a_in_b(const std::vector<std::string>& names1, 
                       const std::vector<std::string>& names2);
std::vector<size_t> a_in_b_index(const std::vector<std::string>& names1, const std::vector<std::string>& names2);

std::vector<bool> a_not_in_b(const std::vector<std::string>& names1, const std::vector<std::string>& names2);

std::vector<size_t> a_not_in_b_index(const std::vector<std::string>& names1, const std::vector<std::string>& names2);

// List operations
py::list resizel(const py::list& x, int n);
py::list orderByname(const py::list& x, 
                   const std::vector<std::string>& names);
py::list removeEmptyElement(const py::list& x);

// Matrix math operations
py::array_t<double> subtractM(py::array_t<double> m, 
                             py::array_t<double> v);
int matchCount(py::array_t<double> m, 
             py::array_t<double> v);

py::dict fisher_test_cpp(py::array_t<double>& x, double conf_level = 0.95);

std::vector<std::string> subCPP(const std::vector<std::string>& pattern, const std::vector<std::string>& replacement, const std::vector<std::string>& x);

std::vector<std::string> char_sort(std::vector<std::string> x, bool dsc);

std::vector<size_t> int_sort(std::vector<size_t> x, bool dsc);

std::vector<double> num_sort(std::vector<double> x, bool dsc);

std::vector<std::string> splitExpression(const std::string& expression, int outputType, bool lowerCase = false);

#endif