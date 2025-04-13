#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <map>
#include <unordered_map>
#include "fbn_utils.h" // Include fbn_utils.h directly
#include "fbn_matrix.h"
#include "fbn_core.h"
#include "fbn_column.h"

namespace py = pybind11;

py::array_t<double> extract_gene_state_from_time_series_cube(
    const std::vector<FBNMatrix>& time_series_cube,
    int temporal) 
{
    // This function extracts gene states from a time series cube, i.e., concatenates all matrices 
    // in the cube with 9s in between.
    // Check input validity
    if (time_series_cube.empty()) {
        throw std::runtime_error("time_series_cube cannot be empty");
    }

    // Get dimensions from first matrix
    auto first_mat = time_series_cube[0].matrix_t();
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
        auto m = mat.matrix_t().request();
        final_cols += m.shape[1];
    }
    final_cols += static_cast<size_t>(temporal) * (time_series_cube.size() - 1);

    // Create result matrix
    std::vector<double> res_data(num_genes * final_cols);
    size_t offset = 0;

    // Process each input matrix
    for (size_t i = 0; i < time_series_cube.size(); ++i) {
        auto current_mat = time_series_cube[i].matrix_t().request();
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

// Python-compatible version of extractGeneStates
FBNMatrix extract_gene_states(
    py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names
) {
    auto buf = state_matrix.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input matrix must be 2D");

    size_t nrows = buf.shape[0];
    size_t ncols = buf.shape[1];

    // Step 1: Use a_in_b_index to find indices of target genes in row names
    std::vector<size_t> r_index = a_in_b_index(target_genes, row_names);

    // Step 2: Sort indices with int_sort (assume it has signature: vector<int> int_sort(vector<int>, bool ascending))
    r_index = int_sort(r_index, false);  // assuming 'false' means ascending

    // Step 3: Create new matrix with selected rows
    py::array_t<double> sub({r_index.size(), ncols});
    auto sub_buf = sub.mutable_unchecked<2>();
    auto mat_buf = state_matrix.unchecked<2>();

    for (size_t i = 0; i < r_index.size(); ++i) {
        int row_idx = r_index[i];
        for (size_t j = 0; j < ncols; ++j) {
            sub_buf(i, j) = mat_buf(row_idx, j);
        }
    }

    std::vector<std::string> col_names;
    for (size_t i = 0; i < ncols; ++i) {
        col_names.push_back(std::to_string(i + 1));
    }
   
    FBNMatrix output = FBNMatrix(sub, target_genes, col_names);
    return output;
}


py::list generate_temporal_gene_states(
    py::dict& main_parameters,
    const std::vector<std::string>& target_gene,
    const std::vector<std::string>& conditional_genes,
    int temporal)
{
    // This function generates temporal gene states based on the provided parameters, and slices the matrices accordingly based 
    // on the target gene and conditional genes.
    // Temporary variables here is referred to how many previous time steps required to determine the current time step
    // Extract parameters from dictionary with validation
    // currentStates means the target gene states
    // previousStates means the conditional gene states
    // currentStates_c means the conditional gene states as target gene states
    // previousStates_c means the target gene states as conditional gene states
    // each state is a list of matrices in different temporal, for eaxample, 1 means the current state is determined by the previous state at 1 step., 
    // 2 means the current state is determined by the previous state at 2 steps.
    if (!main_parameters.contains("currentStates") || 
        !main_parameters.contains("previousStates") ||
        !main_parameters.contains("currentStates_c") || 
        !main_parameters.contains("previousStates_c")) {
        throw std::runtime_error("Missing required parameters in main_parameters");
    }

    // each state is a matrix
    py::list get_current_states = main_parameters["currentStates"].cast<py::list>();
    py::list get_previous_states = main_parameters["previousStates"].cast<py::list>();
    py::list get_current_states_c = main_parameters["currentStates_c"].cast<py::list>();
    py::list get_previous_states_c = main_parameters["previousStates_c"].cast<py::list>();
    std::vector<std::string> get_row_names = main_parameters["rownames"].cast<std::vector<std::string>>();

    if (temporal < 1) temporal = 1;

    // Validate input arrays
    auto validate_list = [](const py::list& lst, const std::string& name, int temporal) {
        if (py::len(lst) < temporal || py::len(lst) == 0) {
            throw std::runtime_error(name + " subscript out of bounds");
        }
    };

    validate_list(get_current_states, "getCurrentStates", temporal);
    validate_list(get_previous_states, "getPreviousStates", temporal);
    validate_list(get_current_states_c, "getCurrentStates_c", temporal);
    validate_list(get_previous_states_c, "getPreviousStates_c", temporal);

    py::list result;
    for (int i = 0; i < temporal; i++) {
        py::array_t<double> current_state = get_current_states[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state = get_previous_states[i].cast<py::array_t<double>>();
        py::array_t<double> current_state_c = get_current_states_c[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state_c = get_previous_states_c[i].cast<py::array_t<double>>();

        //======================Data validation========================
        // get the number of rows and cols
        auto current_state_buf = current_state.request();
        auto previous_state_buf = previous_state.request();
        auto current_state_c_buf = current_state_c.request();
        auto previous_state_c_buf = previous_state_c.request();
        // check if all state matrices have the same number of rows and cols
        if (current_state_buf.ndim != 2 || previous_state_buf.ndim != 2 || 
            current_state_c_buf.ndim != 2 || previous_state_c_buf.ndim != 2) {
            throw std::runtime_error("All state matrices must be 2D");
        }
        if (current_state_buf.shape[0] != previous_state_buf.shape[0] || 
            current_state_buf.shape[0] != current_state_c_buf.shape[0] || 
            current_state_buf.shape[0] != previous_state_c_buf.shape[0]) {
            throw std::runtime_error("All state matrices must have the same number of rows");
        }
        if (current_state_buf.shape[1] != previous_state_buf.shape[1] || 
            current_state_buf.shape[1] != current_state_c_buf.shape[1] || 
            current_state_buf.shape[1] != previous_state_c_buf.shape[1]) {
            throw std::runtime_error("All state matrices must have the same number of columns");
        }
        // Check if the number of columns is greater than temporal + 2
        if (current_state_buf.shape[1] < temporal + 2) {
            throw std::runtime_error("Not enough states for this temporal");
        }
        if (previous_state_buf.shape[1] < temporal + 2) {
            throw std::runtime_error("Not enough states for this temporal");
        }
        if (current_state_c_buf.shape[1] < temporal + 2) {
            throw std::runtime_error("Not enough states for this temporal");
        }
        if (previous_state_c_buf.shape[1] < temporal + 2) {
            throw std::runtime_error("Not enough states for this temporal");
        }

        //=======================End of data validation========================

        
        int n_state = current_state.shape(1) - 1;
        int n_row = current_state.shape(0);
        int start_col = i + 1;
        int end_col = n_state + 1;

        // Create slices with bounds checking
        if (start_col >= end_col) {
            throw std::runtime_error("Invalid column range for slicing");
        }

        // Extract submatrices with proper bounds checking
        py::array_t<double> current_state_sliced = current_state.attr("__getitem__")(
            py::make_tuple(py::slice(0, n_row, 1), py::slice(start_col, end_col, 1))).cast<py::array_t<double>>();
        
        py::array_t<double> previous_state_sliced = previous_state.attr("__getitem__")(
            py::make_tuple(py::slice(0, n_row, 1), py::slice(0, n_state - i, 1))).cast<py::array_t<double>>();
        
        py::array_t<double> current_state_sliced_c = current_state_c.attr("__getitem__")(
            py::make_tuple(py::slice(0, n_row, 1), py::slice(start_col, end_col, 1))).cast<py::array_t<double>>();
        
        py::array_t<double> previous_state_sliced_c = previous_state_c.attr("__getitem__")(
            py::make_tuple(py::slice(0, n_row, 1), py::slice(0, n_state - i, 1))).cast<py::array_t<double>>();

        // debug_str("Before Extracted matrices:");
        // debug_str("Before Extracted previous_state_sliced matrix:");
        // debug_function(previous_state_sliced);
        // debug_str("Before Extracted current_state_sliced matrix:"); 
        // debug_function(current_state_sliced);
        // debug_str("Before Extracted previous_state_sliced_c matrix c:");
        // debug_function(previous_state_sliced_c);
        // debug_str("Before Extracted current_state_sliced_c matrix c:"); 
        // debug_function(current_state_sliced_c);

        // Extract gene states with validation
        FBNMatrix extracted_condition = extract_gene_states(previous_state_sliced, conditional_genes, get_row_names);
        FBNMatrix extracted_target = extract_gene_states(current_state_sliced, target_gene, get_row_names);
        FBNMatrix extracted_condition_c = extract_gene_states(previous_state_sliced_c, target_gene, get_row_names);
        FBNMatrix extracted_target_c = extract_gene_states(current_state_sliced_c, conditional_genes, get_row_names);

        // debug_str("Extracted matrices:");
        // debug_str("Extracted condition matrix:"); // conditional
        // extracted_condition.print();
        // debug_function(extracted_condition.matrix_t());
        // debug_str("Extracted target matrix:"); //target
        // extracted_target.print();
        // debug_function(extracted_target.matrix_t());
        // debug_str("Extracted condition matrix c:");  // target
        // extracted_condition_c.print();
        // debug_function(extracted_condition_c.matrix_t());
        // debug_str("Extracted target matrix c:"); //conditional
        // extracted_target_c.print();
        // debug_function(extracted_target_c.matrix_t());
        // Combine matrices: condition on top and target on bottom
        py::array_t<double> concatenated_matrix = mrbind(extracted_condition.matrix_t(), extracted_target.matrix_t());
        py::array_t<double> concatenated_matrix_c = mrbind(extracted_condition_c.matrix_t(), extracted_target_c.matrix_t());

        // convert to FBNMatrix
        // concatenated new row names based on extracted_condition.row_names() and extracted_target.col_names()
        std::vector<std::string> concatenated_row_names = concatenate_row_names(extracted_condition.row_names(), extracted_target.row_names());
        std::vector<std::string> concatenated_row_names_c = concatenate_row_names(extracted_condition_c.row_names(), extracted_target_c.row_names());


        FBNMatrix concatenated_result = FBNMatrix(concatenated_matrix, concatenated_row_names, extracted_target.col_names());
        FBNMatrix concatenated_result_c = FBNMatrix(concatenated_matrix_c, concatenated_row_names_c, extracted_target_c.col_names());
        // Create subresult dictionary
        py::dict subresult;
        subresult["computation_Matrix"] = concatenated_result;
        subresult["computation_Matrix_c"] = concatenated_result_c;
        subresult["timeStep"] = i + 1;

        result.append(subresult);
    }

    return result;
}

// basic calculation of measures
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
    py::array_t<double>& target_F_cond_F_state) {
    
    int target_T_count = 0;
    int target_F_count = 0;

    // Condition counts
    // print below input for debug
    debug_str("state_cond_T_target_T- start");
    debug_function(m);
    debug_function(mc);
    debug_function(cond_T_target_T_state);
    debug_function(cond_F_target_T_state);
    debug_function(cond_T_target_F_state);
    debug_function(cond_F_target_F_state);
    // rename lenTT to count_cond_T_target_T, 
    // rename lenTF to count_cond_F_target_T, 
    // rename lenFT to count_cond_T_target_F, 
    // rename lenFF to count_cond_F_target_F    
    int count_cond_T_target_T = matchCount(m, cond_T_target_T_state);
    int count_cond_F_target_T = matchCount(m, cond_F_target_T_state);
    int count_cond_T_target_F = matchCount(m, cond_T_target_F_state);
    int count_cond_F_target_F = matchCount(m, cond_F_target_F_state);

    // State vectors
    // rename sateTT to state_cond_T_target_T
    // rename stateTF to state_cond_F_target_T
    // rename stateFT to state_cond_T_target_F
    // rename stateFF to state_cond_F_target_F
    std::vector<double> state_cond_T_target_T = {1.0, 1.0};
    std::vector<double> state_cond_F_target_T = {0.0, 1.0};
    std::vector<double> state_cond_T_target_F = {1.0, 0.0};
    std::vector<double> state_cond_F_target_F = {0.0, 0.0};

    int count_target_T_cond_T = matchCount(mc, target_T_cond_T_state);
    int count_target_F_cond_T = matchCount(mc, target_F_cond_T_state);
    int count_target_T_cond_F = matchCount(mc, target_T_cond_F_state);
    int count_target_F_cond_F = matchCount(mc, target_F_cond_F_state);

    // Create and return result dictionary
    py::dict result;
    result["target_T_count"] = count_cond_T_target_T + count_cond_F_target_T;;
    result["target_F_count"] = count_cond_T_target_F + count_cond_F_target_F;;
    result["target_T_count_c"] = count_target_T_cond_T + count_target_T_cond_F;
    result["target_F_count_c"] = count_target_F_cond_T + count_target_F_cond_F;;
    result["cond_T_count"] = count_cond_T_target_T + count_cond_T_target_F;
    result["cond_F_count"] = count_cond_F_target_T + count_cond_F_target_F;
    result["cond_T_count_c"] = count_target_T_cond_T + count_target_F_cond_T;
    result["cond_F_count_c"] = count_target_T_cond_F + count_target_F_cond_F;
    result["count_cond_T_target_T"] = count_cond_T_target_T;
    result["count_cond_F_target_T"] = count_cond_F_target_T;
    result["count_cond_T_target_F"] = count_cond_T_target_F;
    result["count_cond_F_target_F"] = count_cond_F_target_F;
    result["count_target_T_cond_T"] = count_target_T_cond_T;
    result["count_target_F_cond_T"] = count_target_F_cond_T;
    result["count_target_T_cond_F"] = count_target_T_cond_F;
    result["count_target_F_cond_F"] = count_target_F_cond_F;
    
    return result;
}



// Helper functions to replace Rcpp's all/any
template<typename Container, typename Predicate>
bool all(const Container& c, Predicate p) {
    return std::all_of(c.begin(), c.end(), p);
}

template<typename Container, typename Predicate>
bool any(const Container& c, Predicate p) {
    return std::any_of(c.begin(), c.end(), p);
}

py::dict getGeneProbabilities_basic(py::dict& main_parameters_in_ref,
                                   py::object& fixedgenestate,
                                   std::vector<std::string>& target_gene,
                                   std::vector<std::string>& new_conditional_gene,
                                   int temporal)
{
    // Extract parameters from main_parameters_in_ref
    int total_samples = main_parameters_in_ref["total_samples"].cast<int>();
    std::vector<std::string> rownames = main_parameters_in_ref["rownames"].cast<std::vector<std::string>>();
    int n_timepoints = main_parameters_in_ref["total_timepoints"].cast<int>();
    
    std::vector<std::string> conditional_genes;
    std::vector<std::string> conditional_genes2;
    py::dict cur_fixed_state;
    
    debug_str("step 1");
    if(fixedgenestate.is_none()) {
        conditional_genes = new_conditional_gene;
    } else {
        // fixedgenestate = fixedgenestate_obj.cast<py::dict>();
        // cur_fixed_state = py::cast<py::dict>(fixedgenestate);
        debug_str("fixedgenestate is not None");
        cur_fixed_state = fixedgenestate.cast<py::dict>();
        debug_str("can cast fixedgenestate to dict");
        // Get keys (names) from the dictionary
        for (auto item : cur_fixed_state) {
            conditional_genes.push_back(item.first.cast<std::string>());
        };
        conditional_genes2 = conditional_genes;
        
        // Check if all conditional genes are in rownames
        auto is_in_all_genes = [&](const std::string& gene) {
            return std::find(rownames.begin(), rownames.end(), gene) != rownames.end();
        };
        
        if(!all(conditional_genes, is_in_all_genes)) {
            throw std::runtime_error("All or some part of the conditional genes are not founded in the timeseries cube");
        }
        
        // Check if new_conditional_gene is already in conditional_genes
        auto is_new_gene = [&](const std::string& gene) {
            return gene == new_conditional_gene[0];
        };
        
        if(any(conditional_genes, is_new_gene)) {
            std::vector<size_t> indexes = a_in_b_index(new_conditional_gene, conditional_genes);
            for(size_t i = 0; i < indexes.size(); i++) {
                conditional_genes.erase(conditional_genes.begin() + indexes[i] - i);
            }
            
            std::vector<size_t> indexes2 = a_in_b_index(new_conditional_gene, conditional_genes2);
            for(size_t i = 0; i < indexes2.size(); i++) {
                std::string key = conditional_genes2[indexes2[i]];
                cur_fixed_state.attr("pop")(key);
            }
        } else {
            conditional_genes.push_back(new_conditional_gene[0]);
        }
    }
    debug_str("step 2");
    // py::object copy = py::module_::import("copy").attr("deepcopy");
    py::dict cond_gene_T_states = deep_copy_dict(cur_fixed_state);
    py::dict cond_gene_F_states = deep_copy_dict(cur_fixed_state);
    debug_str("step 2 - initialized cond_gene_T_states and cond_gene_F_states");
    debug_function(cond_gene_T_states);
    debug_function(cond_gene_F_states);

    // Add the new conditional gene to the current fixedgenestate
    // loop through conditional_genes and add them to the dictionary
    for(const auto& gene : conditional_genes) {
        if(!cond_gene_T_states.contains(gene.c_str())) {
            cond_gene_T_states[gene.c_str()] = 1;
        }
        if(!cond_gene_F_states.contains(gene.c_str())) {
            cond_gene_F_states[gene.c_str()] = 0;
        }
    }

    debug_str("step 2 - have value to cond_gene_T_states and cond_gene_F_states");
    debug_function(cond_gene_T_states);
    debug_function(cond_gene_F_states);

    // Get states in order, the order is very important
    std::vector<size_t> indexes3 = a_in_b_index(conditional_genes, rownames);
    std::sort(indexes3.begin(), indexes3.end());
    std::vector<std::string> uniqued_conditional_genes;
    for(size_t idx : indexes3) {
        uniqued_conditional_genes.push_back(rownames[idx]);
    }
    debug_str(join_vector(uniqued_conditional_genes, ","));
    debug_str("step 3");
    int num_of_conditional_genes = static_cast<int>(uniqued_conditional_genes.size());
    
    // print uniqued_conditional_genes for debug
    std:: string sep = ", ";
    debug_str(std::string("cond_gene_T_states: ") + join_vector(uniqued_conditional_genes, sep));
    debug_function(cond_gene_T_states);
    // debug_str("uniqued_conditional_genes: " + join_vector(uniqued_conditional_genes, ","));
    cond_gene_T_states = orderByName(cond_gene_T_states, uniqued_conditional_genes);
    cond_gene_F_states = orderByName(cond_gene_F_states, uniqued_conditional_genes);

    debug_str("reordered");
    debug_function(cond_gene_T_states);
    debug_function(cond_gene_F_states);

    debug_str("step 3.1");
    // Convert dictionary values to vectors
    // state T count and state F count
    // stateTCond and stateFCond are the states of the conditional genes of previous time step
    std::vector<double> stateTCond;
    std::vector<double> stateFCond;
    // load prefixed values and current conditional gene states
    for(const auto& gene : uniqued_conditional_genes) {
        stateTCond.push_back(cond_gene_T_states[gene.c_str()].cast<double>());
        stateFCond.push_back(cond_gene_F_states[gene.c_str()].cast<double>());
    }

    debug_str("state cond");
    debug_str("stateTCond: " + join_double(stateTCond, ","));
    debug_str("stateFCond: " + join_double(stateFCond, ","));


    debug_str("step 4");
    std::vector<double> mTRUE = {1.0};
    std::vector<double> mFALSE = {0.0};
    
    // Prepare state vectors
    std::vector<double> cond_T_target_T_state = concatenate(stateTCond, mTRUE);
    std::vector<double> cond_F_target_T_state = concatenate(stateFCond, mTRUE);
    std::vector<double> cond_T_target_F_state = concatenate(stateTCond, mFALSE);
    std::vector<double> cond_F_target_F_state = concatenate(stateFCond, mFALSE);
    
    debug_str("cond_T_target_T_state " + join_double(cond_T_target_T_state, ","));
    debug_str("cond_F_target_T_state " + join_double(cond_F_target_T_state, ","));
    debug_str("cond_T_target_F_state " + join_double(cond_T_target_F_state, ","));
    debug_str("cond_F_target_F_state " + join_double(cond_F_target_F_state, ","));


    // Prepare counter vectors
    std::vector<double> target_T_cond_T_state = concatenate(mTRUE, stateTCond);
    std::vector<double> target_F_cond_T_state = concatenate(mFALSE, stateTCond);
    std::vector<double> target_T_cond_F_state = concatenate(mTRUE, stateFCond);
    std::vector<double> target_F_cond_F_state = concatenate(mFALSE, stateFCond);

    debug_str("target_T_cond_T_state " + join_double(target_T_cond_T_state, ","));
    debug_str("target_F_cond_T_state " + join_double(target_F_cond_T_state, ","));
    debug_str("target_T_cond_F_state " + join_double(target_T_cond_F_state, ","));
    debug_str("target_F_cond_F_state " + join_double(target_F_cond_F_state, ","));
    debug_str("step 5");
    // Get all combinations of temporal timeseries
    py::list getAllTemporalStates = generate_temporal_gene_states(
        main_parameters_in_ref,
        target_gene,
        conditional_genes,
        temporal
    );
    debug_str("step 6");
    py::dict resultGroup;

    debug_str("step 7");
    for(size_t i = 0; i < getAllTemporalStates.size(); i++) {
        debug_str("step 7.1." + std::to_string(i));
        py::dict temporalState = getAllTemporalStates[i].cast<py::dict>();
        int time_step = temporalState["timeStep"].cast<int>();
        int total_calculated_timepoints = n_timepoints - (total_samples * time_step);

        // check if temporalState["computation_Matrix"] is instance of FBNMatrix
        debug_str("step 7.1.1." + std::to_string(i));
        py::array_t<double> computation_Matrix;
        py::array_t<double> computation_Matrix_c;
        py::object computation_obj = temporalState["computation_Matrix"];
        py::object computation_obj_c = temporalState["computation_Matrix_c"];
        debug_function(computation_obj);
        debug_function(computation_obj_c);

        if (py::isinstance<FBNMatrix>(computation_obj)) {
            computation_Matrix = computation_obj.cast<FBNMatrix>().matrix_t();
        } else {
            computation_Matrix = computation_obj.cast<py::array_t<double>>();
        }

        debug_str("step 7.1.2." + std::to_string(i));
        if(py::isinstance<FBNMatrix>(computation_obj_c)) {
            computation_Matrix_c = computation_obj_c.cast<FBNMatrix>().matrix_t();
        } else {
            computation_Matrix_c = computation_obj_c.cast<py::array_t<double>>();
        }

        // debug_function(computation_Matrix);
        // debug_function(computation_Matrix_c);

        debug_str("step 7.2." + std::to_string(i));
        // Create named array objects to avoid temporary reference issues
        py::array_t<double> stateTCond_array(stateTCond.size(), stateTCond.data());

        py::array_t<double> cond_T_target_T_state_array(cond_T_target_T_state.size(), cond_T_target_T_state.data());
        py::array_t<double> cond_F_target_T_state_array(cond_F_target_T_state.size(), cond_F_target_T_state.data());
        py::array_t<double> cond_T_target_F_state_array(cond_T_target_F_state.size(), cond_T_target_F_state.data());
        py::array_t<double> cond_F_target_F_state_array(cond_F_target_F_state.size(), cond_F_target_F_state.data());
        py::array_t<double> target_T_cond_T_state_array(target_T_cond_T_state.size(), target_T_cond_T_state.data());
        py::array_t<double> target_F_cond_T_state_array(target_F_cond_T_state.size(), target_F_cond_T_state.data());
        py::array_t<double> target_T_cond_F_state_array(target_T_cond_F_state.size(), target_T_cond_F_state.data());
        py::array_t<double> target_F_cond_F_state_array(target_F_cond_F_state.size(), target_F_cond_F_state.data());
        debug_str("step 7.3." + std::to_string(i));
        // print all input arrays for debug
        debug_str("stateTCond_array");
        debug_function(stateTCond_array);
        debug_str("computation_Matrix");
        debug_function(computation_Matrix);
        debug_str("computation_Matrix_c");
        debug_function(computation_Matrix_c);
        debug_str("cond_T_target_T_state_array");
        debug_function(cond_T_target_T_state_array);
        debug_str("cond_F_target_T_state_array");
        debug_function(cond_F_target_T_state_array);
        debug_str("cond_T_target_F_state_array");
        debug_function(cond_T_target_F_state_array);
        debug_str("cond_F_target_F_state_array");
        debug_function(cond_F_target_F_state_array);
        debug_str("target_T_cond_T_state_array");
        debug_function(target_T_cond_T_state_array);
        debug_str("target_F_cond_T_state_array");
        debug_function(target_F_cond_T_state_array);
        debug_str("target_T_cond_F_state_array");
        debug_function(target_T_cond_F_state_array);
        debug_str("target_F_cond_F_state_array");
        debug_function(target_F_cond_F_state_array);

        py::dict result = getBasicMeasures(
            stateTCond_array,
            computation_Matrix,
            computation_Matrix_c,
            cond_T_target_T_state_array,
            cond_F_target_T_state_array,
            cond_T_target_F_state_array,
            cond_F_target_F_state_array,
            target_T_cond_T_state_array,
            target_F_cond_T_state_array,
            target_T_cond_F_state_array,
            target_F_cond_F_state_array
        );
        debug_str("step 7.4." + std::to_string(i));

        if (!result.contains("target_T_count") || !result.contains("target_F_count")) {
            throw std::runtime_error("result missing required keys!");
        }
        
        debug_str("step 7.5." + std::to_string(i));
        result["total_calculated_timepoints"] = total_calculated_timepoints;
        result["num_of_conditional_genes"] = num_of_conditional_genes;
        result["timestep"] = time_step;
        debug_str("step 7.6." + std::to_string(i));
        debug_function(result);
        std::string time_step_str = std::to_string(time_step);
        resultGroup[py::str(time_step_str)] = result;  // Recommended for Unicode safety
    }
    debug_str("finished");
    return resultGroup;
}

PYBIND11_MODULE(fbnnet_core, m) {
    m.def("extract_gene_state_from_time_series_cube", 
          &extract_gene_state_from_time_series_cube,
          "Extract gene states from time series cube",
          py::arg("time_series_cube"),
          py::arg("temporal"));
    m.def("extract_gene_states", 
        &extract_gene_states,
        "Extract specific gene states from a matrix",
        py::arg("state_matrix"),
        py::arg("target_genes"),
        py::arg("row_names"));

    m.def("generate_temporal_gene_states", 
        &generate_temporal_gene_states,
        "Generate temporal gene states",
        py::arg("main_parameters"),
        py::arg("target_gene"),
        py::arg("conditional_genes"),
        py::arg("temporal"));
    // m.def("generate_temporal_gene_states", 
    //     &generate_temporal_gene_states,
    //     "Generate temporal gene states",
    //     py::arg("main_parameters"),
    //     py::arg("target_gene"),
    //     py::arg("conditional_genes"),
    //     py::arg("temporal"));
    m.def("get_basic_measures", &getBasicMeasures, "Calculate basic measures for FBN analysis");
    m.def("getGeneProbabilities_basic", &getGeneProbabilities_basic, "A function to get gene probabilities");
}