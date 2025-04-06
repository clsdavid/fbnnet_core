#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <algorithm>
#include "fbn_utils.h"  // Assuming this contains helpers for Python

namespace py = pybind11;

py::list networkFiltering(py::list res) {
    // Get target genes from dict keys
    py::dict res_dict = res.cast<py::dict>();
    std::vector<std::string> targetgenes;
    for (auto item : res_dict) {
        targetgenes.push_back(item.first.cast<std::string>());
    }

    py::list filtered = res.attr("copy")();  // Create deep copy
    size_t len = targetgenes.size();

    for(size_t i=0; i<len; i++) {
        std::string target = targetgenes[i];
        py::list sub_rule_set = res_dict[target].cast<py::list>();
        std::vector<size_t> filteredIndex;
        std::vector<size_t> indexes;

        size_t cond_len = py::len(sub_rule_set);
        for(size_t j=0; j<cond_len; j++) {
            indexes.push_back(j);
            if(sub_rule_set[j].is_none()) {
                filteredIndex.push_back(j);
                continue;
            }

            py::dict rule = sub_rule_set[j].cast<py::dict>();
            for(size_t k=0; k<cond_len; k++) {
                if(sub_rule_set[k].is_none()) {
                    filteredIndex.push_back(k);
                    continue;
                }

                py::dict rule2 = sub_rule_set[k].cast<py::dict>();
                int numOfInput_rule1 = rule["numOfInput"].cast<int>();
                int numOfInput_rule2 = rule2["numOfInput"].cast<int>();
                
                std::string type_rule1 = rule["type"].cast<std::string>();
                std::string type_rule2 = rule2["type"].cast<std::string>();
                
                std::string timestep_rule1 = rule["timestep"].cast<std::string>();
                std::string timestep_rule2 = rule2["timestep"].cast<std::string>();
                
                std::string input_rule1 = rule["input"].cast<std::string>();
                std::string input_rule2 = rule2["input"].cast<std::string>();

                // Split inputs using helper function
                py::list input1 = splitExpression(input_rule1, 2, false);
                py::list input2 = splitExpression(input_rule2, 2, false);

                // Check subset relationship
                if(is_subset(input1, input2)) {
                    if(numOfInput_rule1 < numOfInput_rule2 && 
                       type_rule1 == type_rule2 &&
                       timestep_rule1 == timestep_rule2) {
                        filteredIndex.push_back(k);
                    }
                }
            }
        }

        // Apply filtering
        py::list new_rules;
        for(auto idx : set_difference(indexes, filteredIndex)) {
            new_rules.append(sub_rule_set[idx]);
        }
        
        filtered[target] = new_rules;
    }

    return filtered;
}

// Helper functions needed in fbn_utils.h:
// - splitExpression: Split string into py::list
// - is_subset: Check if all elements of list1 are in list2
// - set_difference: Return elements in a that aren't in b

PYBIND11_MODULE(fbn_core, m) {
    m.def("networkFiltering", &networkFiltering, "Filters network rules");
    
    // Test case
    m.def("test_network_filtering", []() {
        py::list rules = py::dict(
            "GeneA"_a = py::list({
                py::dict(
                    "numOfInput"_a = 2,
                    "type"_a = "activator",
                    "timestep"_a = "1",
                    "input"_a = "A&B"
                ),
                py::none(),
                py::dict(
                    "numOfInput"_a = 3,
                    "type"_a = "activator",
                    "timestep"_a = "1",
                    "input"_a = "A&B&C"
                )
            })
        );
        
        auto filtered = networkFiltering(rules);
        if(py::len(filtered["GeneA"]) != 1) {
            throw std::runtime_error("Filtering failed");
        }
    });
}