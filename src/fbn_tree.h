#ifndef FBN_TREE_H
#define FBN_TREE_H

#include <pybind11/pybind11.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Function declarations
py::dict getGeneProbabilities_measurements(
    std::vector<std::string>& targetGene,
    py::dict& mainParameters,
    std::vector<std::string>& genes,
    py::object& prefix,
    int temporal = 1,
    bool show_basic_measures = false);

py::dict buildProbabilityTreeOnTargetGene(
    std::vector<std::string>& targetGene,
    py::dict& mainParameters,
    std::vector<std::string>& genes,
    py::object& matchedgenes,
    py::object& matchedexpression,
    int maxK = 4,
    int temporal = 1,
    bool show_basic_measures = false,
    bool findPositiveRegulate = false,
    bool findNegativeRegulate = false);

py::dict process_cube_algorithm(
    std::string& target_gene,
    std::vector<std::string>& conditional_genes,
    int maxK,
    int temporal,
    py::dict& main_parameters,
    py::object& matchedgenes,
    py::object& matchedexpression
);

#endif // FBN_TREE_H
