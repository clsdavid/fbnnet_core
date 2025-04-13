#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "fbn_column.h"

namespace py = pybind11;

PYBIND11_MODULE(fbnnet_column, m) {
    py::class_<FBNColumn>(m, "FBNColumn")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&, const std::vector<double>&>())
        .def_readwrite("row_names", &FBNColumn::row_names)
        .def_readwrite("values", &FBNColumn::values)
        .def("size", &FBNColumn::size)
        .def("get_value", &FBNColumn::get_value)
        .def("__repr__", &FBNColumn::to_string)
        .def("is_none", &FBNColumn::is_none)
        .def("get_row_names", &FBNColumn::get_row_names)
        .def("get_values", &FBNColumn::get_values);
}
