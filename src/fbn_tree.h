#ifndef FBN_TREE_H
#define FBN_TREE_H

#include <pybind11/pybind11.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Function declarations
std::vector<std::string> filterTargetGenesByConditionGenes(
    const std::vector<std::string>& targetGenes,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const py::object& matchedgenes = py::none(),
    int temporal = 1,
    const py::object& targetCounts = py::none());

py::dict getGeneprobabilities_measurements(
    const std::string& targetGene,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const py::object& matchedgenes = py::none(),
    int temporal = 1,
    const py::object& targetCounts = py::none());

py::dict buildProbabilityTreeOnTargetGene(
    const std::string& targetGene,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const py::object& matchedgenes = py::none(),
    const py::object& matchedexpression = py::none(),
    int maxK = 4,
    int temporal = 1,
    const py::object& targetCounts = py::none(),
    bool findPositiveRegulate = false,
    bool findNegativeRegulate = false);

py::dict mineNetworksDirect(
    const std::string& targetGene,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const py::object& matchedgenes = py::none(),
    const py::object& matchedexpression = py::none(),
    int maxK = 4,
    int temporal = 1,
    const py::object& targetCounts = py::none(),
    bool findPositiveRegulate = false,
    bool findNegativeRegulate = false);

py::dict internalloopByWhole2(
    const std::string& target_gene,
    const std::vector<std::string>& conditional_genes,
    int maxK,
    int temporal,
    py::dict mainParameters);

#endif // FBN_TREE_H
