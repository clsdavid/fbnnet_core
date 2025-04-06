#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Include your existing C++ headers here
// #include "fbn_core.h"

namespace py = pybind11;

PYBIND11_MODULE(fbnet_py, m) {
    m.doc() = "Python bindings for the fbnet project";

    // Example: Expose a function
    // m.def("function_name", &function_name, "Description of the function");

    // Example: Expose a class
    // py::class_<ClassName>(m, "ClassName")
    //     .def(py::init<>())
    //     .def("method_name", &ClassName::method_name);
}
