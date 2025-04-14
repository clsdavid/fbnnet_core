#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fbn_matrix.h"

namespace py = pybind11;

PYBIND11_MODULE(fbnnet_matrix, m) {
    py::class_<FBNMatrix>(m, "FBNMatrix")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<double>>&, 
                      const std::vector<std::string>&,
                      const std::vector<std::string>&>(),
             py::arg("matrix"),
             py::arg("row_names") = std::vector<std::string>(),
             py::arg("col_names") = std::vector<std::string>())
        .def(py::init<py::array_t<double>, 
                      const std::vector<std::string>&,
                      const std::vector<std::string>&>(),
             py::arg("matrix"),
             py::arg("row_names") = std::vector<std::string>(),
             py::arg("col_names") = std::vector<std::string>())
        .def("matrix", &FBNMatrix::matrix)
        .def("matrix_t", &FBNMatrix::matrix_t)
        .def("row_names", &FBNMatrix::row_names)
        .def("col_names", &FBNMatrix::col_names)
        .def("num_rows", &FBNMatrix::num_rows)
        .def("num_cols", &FBNMatrix::num_cols)
        .def("print", &FBNMatrix::print, py::arg("precision") = 4)
        .def("__repr__", &FBNMatrix::to_string)
        .def("__str__", &FBNMatrix::to_string);

    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Leshi Chen <chenleshi@hotmail.com>";
}