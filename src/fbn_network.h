#ifndef FBN_NETWORK_H
#define FBN_NETWORK_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Function declaration
py::list networkFiltering(py::list res);

#endif // FBN_NETWORK_H
