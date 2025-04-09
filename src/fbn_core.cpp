#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
// #include "fbn_utils.h"

namespace py = pybind11;

py::array_t<double> extract_gene_state_from_time_series_cube(
    const std::vector<py::array_t<double>>& time_series_cube,
    int temporal) 
{
    // This function extracts gene states from a time series cube, i.e., concatenates all matrices 
    // in the cube with 9s in between.
    // Check input validity
    if (time_series_cube.empty()) {
        throw std::runtime_error("time_series_cube cannot be empty");
    }

    // Get dimensions from first matrix
    auto first_mat = time_series_cube[0];
    // Request buffer info
    auto buf = first_mat.request();
    //extract the number of rows from the shape of the array
    size_t num_genes = buf.shape[0];
    //extract the number of cols from the shape of the array
    size_t num_timepoints = buf.shape[1];  // Kept for potential future use

    // Special case: single matrix
    if (time_series_cube.size() == 1) {
        return first_mat;
    }

    // Create the "nine" matrix filled with 9s
    std::vector<double> nine_matrix(num_genes * static_cast<size_t>(temporal), 9.0);

    // Calculate final matrix dimensions
    size_t final_cols = 0;
    for (const auto& mat : time_series_cube) {
        auto m = mat.request();
        final_cols += m.shape[1];
    }
    final_cols += static_cast<size_t>(temporal) * (time_series_cube.size() - 1);

    // Create result matrix
    std::vector<double> res_data(num_genes * final_cols);
    size_t offset = 0;

    // Process each input matrix
    for (size_t i = 0; i < time_series_cube.size(); ++i) {
        auto current_mat = time_series_cube[i].request();
        double* current_data = static_cast<double*>(current_mat.ptr);
        size_t current_cols = current_mat.shape[1];

        // Copy current matrix data
        for (size_t col = 0; col < current_cols; ++col) {
            for (size_t row = 0; row < num_genes; ++row) {
                res_data[row * final_cols + offset] = current_data[row * current_cols + col];
            }
            offset++;
        }

        // Add nine matrix between matrices (except after last one)
        if (i < time_series_cube.size() - 1) {
            for (int col = 0; col < temporal; ++col) {
                for (size_t row = 0; row < num_genes; ++row) {
                    res_data[row * final_cols + offset] = nine_matrix[row * static_cast<size_t>(temporal) + static_cast<size_t>(col)];
                }
                offset++;
            }
        }
    }

    // Return as numpy array with shape (num_genes, final_cols)
    return py::array_t<double>(
        {static_cast<py::ssize_t>(num_genes), static_cast<py::ssize_t>(final_cols)},
        {static_cast<py::ssize_t>(final_cols * sizeof(double)), static_cast<py::ssize_t>(sizeof(double))},
        res_data.data()
    );
}

// py::array_t<double> extract_gene_states(
//     py::array_t<double>& state_matrix,
//     const std::vector<std::string>& target_genes) 
// {
//     // Get matrix dimensions and data
//     // auto buf = state_matrix.request();
//     // double* data = static_cast<double*>(buf.ptr);
//     // size_t num_samples = buf.shape[1];

//     // Get row names (assuming they're stored in an attribute)
//     // py::object rownames = py::getattr(state_matrix, "rownames", py::none());
//     // if(rownames.is_none()) {
//     //     throw std::runtime_error("State matrix must have rownames attribute");
//     // }
//     // std::vector<std::string> row_names = rownames.cast<std::vector<std::string>>();

//     // Import required functions from fbn_utils
//     // py::module fbn_utils = py::module::import("fbn_utils");
//     // py::function a_in_b_index = fbn_utils.attr("a_in_b_index");
//     // py::function int_sort = fbn_utils.attr("int_sort");

//     // Find matching indices
//     // std::vector<int> r_index_obj = a_in_b_index(target_genes, row_names);
//     // std::vector<int> r_index = int_sort(r_index_obj, false);

//     // // Create output matrix
//     // std::vector<double> sub_data(r_index.size() * num_samples);
//     // std::vector<std::string> sub_row_names;

//     // // Copy selected rows
//     // for (size_t i = 0; i < r_index.size(); ++i) {
//     //     int src_row = r_index[i];
//     //     for (size_t j = 0; j < num_samples; ++j) {
//     //         sub_data[i * num_samples + j] = data[src_row * num_samples + j];
//     //     }
//     //     sub_row_names.push_back(row_names[src_row]);
//     // }

//     // // Create numpy array
//     // py::array_t<double> result({static_cast<py::ssize_t>(r_index.size()), 
//     //                           static_cast<py::ssize_t>(num_samples)},
//     //                          sub_data.data());

//     // // Set row names as an attribute
//     // result.attr("rownames") = py::cast(sub_row_names);

//     // return result;
//     py::array_t<double> result = py::array_t<double>();
//     return result;
// }

PYBIND11_MODULE(fbnnet_core, m) {
    m.def("extract_gene_state_from_time_series_cube", 
          &extract_gene_state_from_time_series_cube,
          "Extract gene states from time series cube",
          py::arg("time_series_cube"),
          py::arg("temporal"));
    // m.def("extract_gene_states", 
    //         &extract_gene_states,
    //         "Extract specific gene states from a matrix",
    //         py::arg("state_matrix"),
    //         py::arg("target_genes"),
    //         py::arg("fbn_utils"));
}
// // -------------------- Core Implementation --------------------
// py::array_t<double> extractGeneStateFromTimeSeriesCube(py::list timeSeriesCube, int temporal) {
//     int numOfmats = py::len(timeSeriesCube);
//     if(numOfmats == 1) return timeSeriesCube[0].cast<py::array_t<double>>();
    
//     auto first_mat = timeSeriesCube[0].cast<py::array_t<double>>();
//     auto genes = get_row_names(first_mat); // Assume helper function in fbn_utils
    
//     py::array_t<double> nine_matrix({genes.size(), (size_t)temporal}, 9.0);
//     int final_len = 0;
    
//     // Calculate final length
//     for(auto& item : timeSeriesCube) {
//         auto mat = item.cast<py::array_t<double>>();
//         final_len += mat.shape(1);
//     }
    
//     py::array_t<double> res({genes.size(), (size_t)(final_len + temporal*(numOfmats-1))});
//     int offset = 0;
    
//     for(int i=0; i<numOfmats; i++) {
//         auto mat = timeSeriesCube[i].cast<py::array_t<double>>();
//         auto buf = mat.request();
//         auto cols = mat.shape(1);
        
//         // Copy matrix columns
//         for(int j=0; j<cols; j++) {
//             py::slice col_slice(offset, offset+1, 1);
//             res[py::ellipsis(), col_slice] = mat[py::ellipsis(), j];
//             offset++;
//         }
        
//         // Add 9s matrix between matrices
//         if(i < numOfmats-1) {
//             for(int j=0; j<temporal; j++) {
//                 py::slice col_slice(offset, offset+1, 1);
//                 res[py::ellipsis(), col_slice] = nine_matrix[py::ellipsis(), j];
//                 offset++;
//             }
//         }
//     }
    
//     set_row_names(res, genes); // Helper to preserve row names
//     return res;
// }

// // -------------------- extractGeneStates --------------------
// py::array_t<double> extractGeneStates(
//   py::array_t<double> stateMatrix,
//   const std::vector<std::string>& targetgenes) 
// {
//   // Get row names from matrix attribute
//   auto row_names = get_row_names(stateMatrix);
  
//   // Find indices of target genes
//   auto r_index = a_in_b_index(targetgenes, row_names);
  
//   // Sort indices in ascending order
//   std::sort(r_index.begin(), r_index.end());
  
//   // Create output matrix
//   py::array_t<double> sub({(size_t)r_index.size(), (size_t)stateMatrix.shape(1)});
//   auto sub_buf = sub.mutable_unchecked<2>();
//   auto state_buf = stateMatrix.unchecked<2>();
  
//   // Copy selected rows
//   for(size_t i=0; i<r_index.size(); i++) {
//       if(r_index[i] >= stateMatrix.shape(0)) {
//           throw std::out_of_range("Row index out of bounds");
//       }
//       for(size_t j=0; j<stateMatrix.shape(1); j++) {
//           sub_buf(i, j) = state_buf(r_index[i], j);
//       }
//   }
  
//   // Set row names for output matrix
//   std::vector<std::string> selected_names;
//   for(auto idx : r_index) {
//       selected_names.push_back(row_names[idx]);
//   }
//   set_row_names(sub, selected_names);
  
//   return sub;
// }

// // -------------------- generateTemporalGeneStates --------------------
// py::list generateTemporalGeneStates(
//   py::dict mainParameters,
//   const std::vector<std::string>& targetgene,
//   const std::vector<std::string>& conditional_genes,
//   int temporal) 
// {
//   if(temporal < 1) temporal = 1;
  
//   // Extract parameter lists
//   auto getCurrentStates = mainParameters["currentStates"].cast<py::list>();
//   auto getpreviousStates = mainParameters["previousStates"].cast<py::list>();
//   auto getCurrentStates_c = mainParameters["currentStates_c"].cast<py::list>();
//   auto getpreviousStates_c = mainParameters["previousStates_c"].cast<py::list>();
  
//   // Validate list lengths
//   auto validate = [temporal](const py::list& lst, const std::string& name) {
//       if(py::len(lst) < temporal || py::len(lst) == 0) {
//           throw std::runtime_error(name + " subscript out of bounds");
//       }
//   };
  
//   validate(getCurrentStates, "getCurrentStates");
//   validate(getpreviousStates, "getpreviousStates");
//   validate(getCurrentStates_c, "getCurrentStates_c");
//   validate(getpreviousStates_c, "getpreviousStates_c");
  
//   py::list result(temporal);
  
//   for(int i=0; i<temporal; i++) {
//       // Get current time step matrices
//       auto currentState = getCurrentStates[i].cast<py::array_t<double>>();
//       auto previousState = getpreviousStates[i].cast<py::array_t<double>>();
//       auto currentState_c = getCurrentStates_c[i].cast<py::array_t<double>>();
//       auto previousState_c = getpreviousStates_c[i].cast<py::array_t<double>>();
      
//       // Validate column counts
//       int n_state = previousState.shape(1);
//       if((n_state - temporal) < 2) {
//           throw std::runtime_error("Not enough states for this temporal");
//       }
//       n_state--;  // Adjust for 0-based indexing
      
//       // Slice columns using py::slice
//       auto slice_cols = [](py::array_t<double> mat, int start, int end) {
//           return mat[py::ellipsis(), py::slice(start, end, 1)];
//       };
      
//       // Process current state
//       auto t_current = slice_cols(currentState, i+1, n_state+1);
//       auto t_previous = slice_cols(previousState, 0, n_state - i);
      
//       // Process counter state
//       auto t_current_c = slice_cols(currentState_c, i+1, n_state+1);
//       auto t_previous_c = slice_cols(previousState_c, 0, n_state - i);
      
//       // Extract relevant gene states
//       auto extractedCondition = extractGeneStates(t_previous, conditional_genes);
//       auto extractedTarget = extractGeneStates(t_current, targetgene);
//       auto extractedCondition_c = extractGeneStates(t_previous_c, targetgene);
//       auto extractedTarget_c = extractGeneStates(t_current_c, conditional_genes);
      
//       // Row-bind matrices
//       auto concatenatedMatrix = mrbind(extractedCondition, extractedTarget);
//       auto concatenatedMatrix_c = mrbind(extractedCondition_c, extractedTarget_c);
      
//       // Build subresult
//       py::dict subresult;
//       subresult["computation_Matrix"] = concatenatedMatrix;
//       subresult["computation_Matrix_c"] = concatenatedMatrix_c;
//       subresult["timeStep"] = i+1;
      
//       result[i] = subresult;
//   }
  
//   return result;
// }



// // -------------------- getBasicMeasures --------------------
// py::dict getBasicMeasures(
//   py::array_t<double> stateTCond,
//   py::array_t<double> m,
//   py::array_t<double> mc,
//   py::array_t<double> cond_T_target_T_state,
//   py::array_t<double> cond_F_target_T_state,
//   py::array_t<double> cond_T_target_F_state,
//   py::array_t<double> cond_F_target_F_state,
//   py::array_t<double> cond_T_target_T_state_c,
//   py::array_t<double> cond_F_target_T_state_c,
//   py::array_t<double> cond_T_target_F_state_c,
//   py::array_t<double> cond_F_target_F_state_c,
//   bool recount_target) 
// {
//   auto matchCount = [](py::array_t<double> matrix, py::array_t<double> pattern) {
//       // Implementation needed in fbn_utils.h
//   };

//   int target_T_count = 0;
//   int target_F_count = 0;

//   // Calculate match counts
//   int lenTT = matchCount(m, cond_T_target_T_state);
//   int lenTF = matchCount(m, cond_F_target_T_state);
//   int lenFT = matchCount(m, cond_T_target_F_state);
//   int lenFF = matchCount(m, cond_F_target_F_state);

//   if(recount_target) {
//       if(stateTCond.size() > 1) {
//           auto m2 = m[py::slice(-2, py::none(), 1), py::all()];
//           int sresTT = matchCount(m2, py::array_t<double>({1.0, 1.0}));
//           int sresTF = matchCount(m2, py::array_t<double>({0.0, 1.0}));
//           int sresFT = matchCount(m2, py::array_t<double>({1.0, 0.0}));
//           int sresFF = matchCount(m2, py::array_t<double>({0.0, 0.0}));
          
//           target_T_count = sresTT + sresTF;
//           target_F_count = sresFT + sresFF;
//       } else {
//           target_T_count = lenTT + lenTF;
//           target_F_count = lenFT + lenFF;
//       }
//   }

//   // Calculate counter counts
//   int lenTT_c = matchCount(mc, cond_T_target_T_state_c);
//   int lenTF_c = matchCount(mc, cond_F_target_T_state_c);
//   int lenFT_c = matchCount(mc, cond_T_target_F_state_c);
//   int lenFF_c = matchCount(mc, cond_F_target_F_state_c);

//   return py::dict(
//       "target_T_count"_a = target_T_count,
//       "target_F_count"_a = target_F_count,
//       "cond_T_count"_a = lenTT + lenFT,
//       "cond_F_count"_a = lenTF + lenFF,
//       "cond_T_count_c"_a = lenTT_c + lenFT_c,
//       "cond_F_count_c"_a = lenTF_c + lenFF_c,
//       "lenTT"_a = lenTT,
//       "lenTF"_a = lenTF,
//       "lenFT"_a = lenFT,
//       "lenFF"_a = lenFF,
//       "lenTT_c"_a = lenTT_c,
//       "lenTF_c"_a = lenTF_c,
//       "lenFT_c"_a = lenFT_c,
//       "lenFF_c"_a = lenFF_c
//   );
// }

// // -------------------- getGeneProbabilities_basic --------------------
// py::list getGeneProbabilities_basic(
//   py::dict main_parameters_in_ref,
//   const py::object& fixedgenestate,
//   const std::vector<std::string>& target_gene,
//   const std::vector<std::string>& new_conditional_gene,
//   int temporal,
//   const py::object& targetCounts = py::none()) 
// {
//   // Helper functions needed in fbn_utils.h:
//   // - a_in_b_index
//   // - concatenate
//   // - resizel
//   // - orderByname

//   auto dataCube = main_parameters_in_ref["timeseries"].cast<py::list>();
//   auto all_gene_names = get_row_names(dataCube[0].cast<py::array_t<double>>());
  
//   std::vector<std::string> conditional_genes;
//   py::list cur_fixed_state;

//   if(fixedgenestate.is_none()) {
//       conditional_genes = new_conditional_gene;
//   } else {
//       cur_fixed_state = fixedgenestate.cast<py::list>();
//       conditional_genes = get_list_names(cur_fixed_state);
      
//       // Validate genes exist
//       if(!all(a_in_b(conditional_genes, all_gene_names))) {
//           throw std::runtime_error("Some conditional genes not found");
//       }

//       // Handle overlapping genes
//       auto overlap = a_in_b_index(new_conditional_gene, conditional_genes);
//       if(!overlap.empty()) {
//           // Remove overlapping elements
//           // Implementation depends on list manipulation helpers
//       } else {
//           conditional_genes = concatenate(conditional_genes, new_conditional_gene);
//       }
//   }

//   // Process state vectors
//   py::list cond_gene_T_states = process_states(cur_fixed_state, 1);
//   py::list cond_gene_F_states = process_states(cur_fixed_state, 0);
  
//   // Generate temporal states
//   auto getAllTemporalStates = generateTemporalGeneStates(
//       main_parameters_in_ref, 
//       target_gene,
//       conditional_genes,
//       temporal
//   );

//   // Process results
//   py::list resultGroup;
//   bool recount_target = targetCounts.is_none();
//   py::list new_targetCounts;

//   for(auto& temporalState : getAllTemporalStates) {
//       auto basic = getBasicMeasures(
//           // ... parameters ...
//       );
      
//       if(recount_target) {
//           py::dict targets;
//           targets["target_T_count"] = basic["target_T_count"];
//           targets["target_F_count"] = basic["target_F_count"];
//           new_targetCounts.append(targets);
//       }
      
//       // Add additional metrics
//       basic["total_calculated_timepoints"] = calculate_timepoints(dataCube);
//       basic["num_of_conditional_genes"] = conditional_genes.size();
//       basic["timestep"] = temporalState["timeStep"];
      
//       resultGroup.append(basic);
//   }
  
//   return resultGroup;
// }

// // -------------------- getAdvancedMeasures --------------------
// py::dict getAdvancedMeasures(const py::dict& basic_measures) {
//   // Extract basic measures
//   auto extract = [&](const char* name) { return basic_measures[name].cast<double>(); };
  
//   double cond_T_count = extract("cond_T_count");
//   double cond_F_count = extract("cond_F_count");
//   double target_T_count = extract("target_T_count");
//   double target_F_count = extract("target_F_count");
//   double lenTT = extract("lenTT");
//   double lenTF = extract("lenTF");
//   // ... extract all needed values ...

//   // Calculate supports
//   double total_timepoints = extract("total_calculated_timepoints");
//   auto dround = [](double val, int dec) { return std::round(val * std::pow(10, dec)) / std::pow(10, dec); };
  
//   double condition_T_support = dround(cond_T_count / total_timepoints, 5);
//   double target_T_support = dround(target_T_count / total_timepoints, 5);
//   // ... other supports ...

//   // Calculate confidence values
//   double confidence_TT = cond_T_count > 0 ? dround(lenTT / cond_T_count, 5) : 0;
//   // ... other confidence calculations ...

//   // Calculate causality tests
//   double causality_test_TT = 99999;
//   if(counter_confidence_TT != 0) {
//       causality_test_TT = dround(confidence_TT / counter_confidence_TT, 2);
//   }
//   // ... other causality tests ...

//   // Build final result dict
//   py::dict result;
//   result["TT"] = confidence_TT;
//   result["TF"] = confidence_TF;
//   // ... add all 42 metrics ...
//   result["bestFitP"] = calculate_best_fit(signal_activator, ...);
//   result["bestFitN"] = calculate_best_fit(signal_inhibitor, ...);
  
//   return result;
// }

// // -------------------- getGeneProbabilities_advanced --------------------
// py::dict getGeneProbabilities_advanced(const py::list& getGeneProbabilities_basic) {
//   int len = py::len(getGeneProbabilities_basic);
//   py::list resultGroup(len);
//   py::list targetCounts(len);

//   py::dict bestFitP;
//   py::dict bestFitN;
//   bool first = true;

//   for(int j=0; j<len; j++) {
//       py::dict basic = getGeneProbabilities_basic[j].cast<py::dict>();
//       py::dict advanced = getAdvancedMeasures(basic);
      
//       // Store advanced measures
//       resultGroup[j] = advanced;

//       // Create target counts entry
//       py::dict targets;
//       targets["target_T_count"] = basic["target_T_count"].cast<int>();
//       targets["target_F_count"] = basic["target_F_count"].cast<int>();
//       targetCounts[j] = targets;

//       // Find best fits
//       if(first) {
//           bestFitP = advanced;
//           bestFitN = advanced;
//           first = false;
//           continue;
//       }

//       double current_bestP = bestFitP["bestFitP"].cast<double>();
//       double current_bestN = bestFitN["bestFitN"].cast<double>();
//       double new_bestP = advanced["bestFitP"].cast<double>();
//       double new_bestN = advanced["bestFitN"].cast<double>();
//       int current_stepP = bestFitP["timestep"].cast<int>();
//       int current_stepN = bestFitN["timestep"].cast<int>();
//       int new_step = advanced["timestep"].cast<int>();

//       // Update best P
//       if((new_bestP < current_bestP) || 
//          (new_bestP == current_bestP && new_step < current_stepP)) {
//           bestFitP = advanced;
//       }

//       // Update best N
//       if((new_bestN < current_bestN) || 
//          (new_bestN == current_bestN && new_step < current_stepN)) {
//           bestFitN = advanced;
//       }
//   }

//   return py::dict(
//       "getBestFitP"_a = bestFitP,
//       "getBestFitN"_a = bestFitN,
//       "targetCounts"_a = targetCounts
//   );
// }

// // -------------------- getGenePrababilities --------------------
// py::dict getGeneProbabilities(
//   py::dict main_parameters_in_ref,
//   const py::object& fixedgenestate,
//   const std::vector<std::string>& target_gene,
//   const std::vector<std::string>& new_conditional_gene,
//   int temporal,
//   const py::object& targetCounts = py::none()) 
// {
//   // Get basic measures
//   py::list basic_measures = getGeneProbabilities_basic(
//       main_parameters_in_ref,
//       fixedgenestate,
//       target_gene,
//       new_conditional_gene,
//       temporal,
//       targetCounts
//   );

//   // Get advanced probabilities
//   py::dict probability = getGeneProbabilities_advanced(basic_measures);

//   if(probability.is_none()) {
//       return py::dict();
//   }

//   return py::dict(
//       "getBestFitP"_a = probability["getBestFitP"],
//       "getBestFitN"_a = probability["getBestFitN"],
//       "targetCounts"_a = probability["targetCounts"]
//   );
// }


// // -------------------- Test Cases --------------------
// PYBIND11_MODULE(fbn_core, m) {
//   m.def("extractGeneStateFromTimeSeriesCube", &extractGeneStateFromTimeSeriesCube,
//       py::arg("timeSeriesCube"), py::arg("temporal") = 1);

//   // Test case 1: Basic functionality
//   m.def("test_extractGeneState", []() {
//       py::list cubes;
//       cubes.append(py::array_t<double>({{1,2}, {3,4}})); // 2x2
//       cubes.append(py::array_t<double>({{5,6}, {7,8}})); // 2x2
      
//       auto result = extractGeneStateFromTimeSeriesCube(cubes, 1);
//       py::print("Test 1 Result:\n", result);
      
//       // Verify dimensions
//       if(result.shape(1) != 2+2+1) throw std::runtime_error("Test 1 failed");
//   });
  
//   // Similar test cases for other functions
//   m.def("test_all", []() {
//       test_extractGeneState();
//       // Add other test calls
//   });

//   m.def("extractGeneStates", &extractGeneStates,
//     py::arg("stateMatrix"), py::arg("targetgenes"));
    
//   m.def("generateTemporalGeneStates", &generateTemporalGeneStates,
//       py::arg("mainParameters"), py::arg("targetgene"), 
//       py::arg("conditional_genes"), py::arg("temporal") = 1);

//   m.def("test_extractGeneStates", []() {
//       // Create test matrix with row names
//       py::array_t<double> mat({3, 2}, {1,2,3,4,5,6});
//       set_row_names(mat, {"gene1", "gene2", "gene3"});
      
//       // Test extraction
//       auto sub = extractGeneStates(mat, {"gene3", "gene1"});
//       py::print("Extracted matrix:\n", sub);
      
//       // Verify output
//       if(sub.shape(0) != 2) throw std::runtime_error("Test 1 failed");
//       auto names = get_row_names(sub);
//       if(names != std::vector<std::string>{"gene1", "gene3"}) 
//           throw std::runtime_error("Row name order incorrect");
//   });

//   m.def("test_generateTemporal", []() {
//       py::dict params;
//       // Populate test data
//       py::list currentStates, previousStates;
//       // ... create sample matrices ...
//       params["currentStates"] = currentStates;
//       params["previousStates"] = previousStates;
//       params["currentStates_c"] = currentStates;
//       params["previousStates_c"] = previousStates;
      
//       auto result = generateTemporalGeneStates(params, {"target"}, {"cond"}, 1);
//       py::print("Temporal result:", result);
      
//       if(py::len(result) != 1) 
//           throw std::runtime_error("Temporal test failed");
//   });
//   m.def("extractGeneStates", &extractGeneStates,
//     py::arg("stateMatrix"), py::arg("targetgenes"));
    
//   m.def("generateTemporalGeneStates", &generateTemporalGeneStates,
//       py::arg("mainParameters"), py::arg("targetgene"), 
//       py::arg("conditional_genes"), py::arg("temporal") = 1);

//   m.def("test_extractGeneStates", []() {
//       // Create test matrix with row names
//       py::array_t<double> mat({3, 2}, {1,2,3,4,5,6});
//       set_row_names(mat, {"gene1", "gene2", "gene3"});
      
//       // Test extraction
//       auto sub = extractGeneStates(mat, {"gene3", "gene1"});
//       py::print("Extracted matrix:\n", sub);
      
//       // Verify output
//       if(sub.shape(0) != 2) throw std::runtime_error("Test 1 failed");
//       auto names = get_row_names(sub);
//       if(names != std::vector<std::string>{"gene1", "gene3"}) 
//           throw std::runtime_error("Row name order incorrect");
//   });

//   m.def("test_generateTemporal", []() {
//       py::dict params;
//       // Populate test data
//       py::list currentStates, previousStates;
//       // ... create sample matrices ...
//       params["currentStates"] = currentStates;
//       params["previousStates"] = previousStates;
//       params["currentStates_c"] = currentStates;
//       params["previousStates_c"] = previousStates;
      
//       auto result = generateTemporalGeneStates(params, {"target"}, {"cond"}, 1);
//       py::print("Temporal result:", result);
      
//       if(py::len(result) != 1) 
//           throw std::runtime_error("Temporal test failed");
//   });
//   m.def("getBasicMeasures", &getBasicMeasures,
//     py::arg("stateTCond"), py::arg("m"), py::arg("mc"),
//     py::arg("cond_T_target_T_state"), py::arg("cond_F_target_T_state"),
//     py::arg("cond_T_target_F_state"), py::arg("cond_F_target_F_state"),
//     py::arg("cond_T_target_T_state_c"), py::arg("cond_F_target_T_state_c"),
//     py::arg("cond_T_target_F_state_c"), py::arg("cond_F_target_F_state_c"),
//     py::arg("recount_target"));

//   m.def("getGenePrababilities_basic", &getGenePrababilities_basic,
//       py::arg("main_parameters_in_ref"), py::arg("fixedgenestate") = py::none(),
//       py::arg("target_gene"), py::arg("new_conditional_gene"),
//       py::arg("temporal"), py::arg("targetCounts") = py::none());

//   m.def("getAdvancedMeasures", &getAdvancedMeasures,
//       py::arg("basic_measures"));

//   m.def("test_basic_measures", []() {
//       py::array_t<double> m({{1,1}, {0,1}, {1,0}});
//       py::array_t<double> pattern({1,1});
//       auto counts = matchCount(m, pattern);
//       if(counts != 1) throw std::runtime_error("Match count failed");
//   });

//   m.def("getGenePrababilities", &getGenePrababilities,
//     py::arg("main_parameters_in_ref"),
//     py::arg("fixedgenestate") = py::none(),
//     py::arg("target_gene"),
//     py::arg("new_conditional_gene"),
//     py::arg("temporal"),
//     py::arg("targetCounts") = py::none());

//   m.def("test_advanced_probabilities", []() {
//       py::list basic_measures;
//       // Create test basic measures
//       // ...
      
//       auto result = getGenePrababilities_advanced(basic_measures);
//       if(result.size() == 0) throw std::runtime_error("Empty result");
//   });

// }