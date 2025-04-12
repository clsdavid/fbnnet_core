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
    py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names);

py::list generate_temporal_gene_states(
    py::dict& main_parameters,
    const std::vector<std::string>& target_gene,
    const std::vector<std::string>& conditional_genes,
    int temporal);

py::dict createResultDict(int target_T_count, int target_F_count,
    int cond_T_count, int cond_F_count,
    int cond_T_count_c, int cond_F_count_c,
    int lenTT, int lenTF, int lenFT, int lenFF,
    int lenTT_c, int lenTF_c, int lenFT_c, int lenFF_c);

py::dict getBasicMeasures(
    py::array_t<double>& stateTCond,
    py::array_t<double>& m,
    py::array_t<double>& mc,
    py::array_t<double>& cond_T_target_T_state,
    py::array_t<double>& cond_F_target_T_state,
    py::array_t<double>& cond_T_target_F_state,
    py::array_t<double>& cond_F_target_F_state,
    py::array_t<double>& cond_T_target_T_state_c,
    py::array_t<double>& cond_F_target_T_state_c,
    py::array_t<double>& cond_T_target_F_state_c,
    py::array_t<double>& cond_F_target_F_state_c,
    bool recount_target
);
py::dict getGenePrababilities_basic(py::dict& main_parameters_in_ref,
    py::object& fixedgenestate,
    std::vector<std::string>& target_gene,
    std::vector<std::string>& new_conditional_gene,
    int temporal,
    py::object& targetCounts);
#endif
