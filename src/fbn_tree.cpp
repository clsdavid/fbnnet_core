#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include "fbn_core.h"
#include "fbn_tree.h"
#include "fbn_utils.h"

namespace py = pybind11;


// Main function
py::dict getGeneProbabilities_measurements(
    std::vector<std::string>& targetGene,
    py::dict& mainParameters,
    std::vector<std::string>& genes,
    py::object& prefix,
    int temporal,
    bool show_basic_measures)
{
    size_t len = genes.size();
    py::dict result_dict;
    for(size_t i = 0; i < len; i++) {
        std::vector<std::string> gene = {genes[i]};
        py::print("Processing gene:", genes[i]);

        py::dict probabilityOfFourCombines = getGeneProbabilities(
            mainParameters,
            prefix,
            targetGene,
            gene,
            temporal,
            show_basic_measures
        );

        py::print("probabilityOfFourCombines:", probabilityOfFourCombines);
        if(probabilityOfFourCombines.is_none()) {
            continue;
        }

        py::dict probabilityOfFourCombines_P;
        py::dict probabilityOfFourCombines_N;

        probabilityOfFourCombines_P = probabilityOfFourCombines["BestFitP"];
        probabilityOfFourCombines_N = probabilityOfFourCombines["BestFitN"];

        // Check essential gene conditions
        if(!probabilityOfFourCombines_P["is_essential_gene"].cast<bool>() &&
           !probabilityOfFourCombines_N["is_essential_gene"].cast<bool>()) {
            continue;
        }

        if(!probabilityOfFourCombines_P["is_essential_gene"].cast<bool>()) {
            probabilityOfFourCombines_P = probabilityOfFourCombines_N;
        }

        if(!probabilityOfFourCombines_N["is_essential_gene"].cast<bool>()) {
            probabilityOfFourCombines_N = probabilityOfFourCombines_P;
        }

        // Check correlation conditions
        if(!probabilityOfFourCombines_P["isPositiveCorrelated"].cast<bool>() &&
           !probabilityOfFourCombines_P["isNegativeCorrelated"].cast<bool>()) {
            probabilityOfFourCombines_P = probabilityOfFourCombines_N;
        }

        if(!probabilityOfFourCombines_N["isPositiveCorrelated"].cast<bool>() &&
           !probabilityOfFourCombines_N["isNegativeCorrelated"].cast<bool>()) {
            
            bool all_equal = (probabilityOfFourCombines_P.equal(probabilityOfFourCombines_N));
            if(all_equal) {
                continue;
            }
            probabilityOfFourCombines_N = probabilityOfFourCombines_P;
        }

        // Prepare the result dictionary
        py::dict gene_result;
        gene_result["probabilityOfFourCombines_P"] = probabilityOfFourCombines_P;
        gene_result["probabilityOfFourCombines_N"] = probabilityOfFourCombines_N;

        // Add to main result dictionary with gene name as key
        result_dict[genes[i].c_str()] = gene_result;
    }

    // Set names attribute (equivalent to R's names(res) <- names)
    return result_dict;
}

// PyBind11 module definition
PYBIND11_MODULE(fbnnet_tree, m) {
    m.def("getGeneProbabilities_measurements", &getGeneProbabilities_measurements,
        "Get the main measurements based on the input data",
        py::arg("targetGene"),
        py::arg("mainParameters"),
        py::arg("genes"),
        py::arg("prefix") = py::none(),
        py::arg("temporal") = 1,
        py::arg("show_basic_measures") = false);
}