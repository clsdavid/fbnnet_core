#ifndef FBN_CORE_H
#define FBN_CORE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <algorithm>
#include "fbn_utils.h"
#include "fbn_core.h"

namespace py = pybind11;

py::array_t<double> extractGeneStateFromTimeSeriesCube(
    const std::vector<py::array_t<double>>& timeSeriesCube,
    int temporal);
    
py::dict extract_gene_states(
    py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names);

#endif