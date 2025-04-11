#ifndef FBN_CORE_H
#define FBN_CORE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <algorithm>

#include "fbn_utils.h"
#include "fbn_matrix.h"

namespace py = pybind11;

py::array_t<double> extractGeneStateFromTimeSeriesCube(
    const std::vector<py::array_t<double>>& timeSeriesCube,
    int temporal);

FBNMatrix extract_gene_states(
    const py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names);

py::list generate_temporal_gene_states(
    const py::dict& main_parameters,
    const std::vector<std::string>& target_gene,
    const std::vector<std::string>& conditional_genes,
    int temporal);

#endif
