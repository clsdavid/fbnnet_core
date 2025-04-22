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
    int count_cond_T_target_T, int count_cond_F_target_T, int count_cond_T_target_F, int count_cond_F_target_F,
    int count_target_T_cond_T, int count_target_F_cond_T, int count_target_T_cond_F, int count_target_F_cond_F);

py::dict getBasicMeasures(
    py::array_t<double>& stateTCond,
    py::array_t<double>& m,
    py::array_t<double>& mc,
    py::array_t<double>& cond_T_target_T_state,
    py::array_t<double>& cond_F_target_T_state,
    py::array_t<double>& cond_T_target_F_state,
    py::array_t<double>& cond_F_target_F_state,
    py::array_t<double>& target_T_cond_T_state,
    py::array_t<double>& target_F_cond_T_state,
    py::array_t<double>& target_T_cond_F_state,
    py::array_t<double>& target_F_cond_F_state
);
py::dict getGeneProbabilities_basic(py::dict& main_parameters_in_ref,
    py::object& fixedgenestate,
    std::vector<std::string>& target_gene,
    std::vector<std::string>& new_conditional_gene,
    int temporal);

py::dict getAdvancedMeasures(py::dict& basic_measures, bool show_basic_measures = false);
py::dict getGeneProbabilities_advanced(py::dict& geneProbabilities_basic, bool show_basic_measures = false);
py::dict getGeneProbabilities(py::dict& data,
    py::object& prefix,
    std::vector<std::string>& target,
    std::vector<std::string>& condition,
    int temporal,
    bool show_basic_measures = false);
#endif
