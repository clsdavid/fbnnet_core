#ifndef FBN_CORE_H
#define FBN_CORE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> extractGeneStateFromTimeSeriesCube(
    py::list timeSeriesCube,
    int temporal = 1
);

py::array_t<double> extractGeneStates(
    py::array_t<double> stateMatrix,
    std::vector<std::string> targetgenes
);

py::list generateTemporalGeneStates(
    py::dict mainParameters,
    std::vector<std::string> targetgene,
    std::vector<std::string> conditional_genes,
    int temporal = 1
);

py::list getBasicMeasures(
    py::array_t<double> stateTCond,
    py::array_t<double> m,
    py::array_t<double> mc,
    py::array_t<double> cond_T_target_T_state,
    py::array_t<double> cond_F_target_T_state,
    py::array_t<double> cond_T_target_F_state,
    py::array_t<double> cond_F_target_F_state,
    py::array_t<double> cond_T_target_T_state_c,
    py::array_t<double> cond_F_target_T_state_c,
    py::array_t<double> cond_T_target_F_state_c,
    py::array_t<double> cond_F_target_F_state_c,
    bool recount_target
);

py::list getAdvancedMeasures(const py::list& basic_measures);

py::list getGenePrababilities_advanced(const py::list& getGenePrababilities_basic);

py::list getGenePrababilities_basic(
    py::dict main_parameters_in_ref,
    const py::object& fixedgenestate,
    std::vector<std::string> target_gene,
    std::vector<std::string> new_conditional_gene,
    int temporal = 1,
    const py::object& targetCounts = py::none()
);

py::list getGenePrababilities(
    py::dict main_parameters_in_ref,
    const py::object& fixedgenestate,
    std::vector<std::string> target_gene,
    std::vector<std::string> new_conditional_gene,
    int temporal = 1,
    const py::object& targetCounts = py::none()
);

#endif