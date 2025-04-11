#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include "fbn_utils.h" // Include fbn_utils.h directly
#include "fbn_matrix.h"

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

FBNMatrix extract_gene_states(
    py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names) 
{
    // Get matrix dimensions and data
    auto buf = state_matrix.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t num_genes = buf.shape[0];
    size_t num_time_points = buf.shape[1];


    // Check dimensions match
    if (row_names.size() != num_genes) {
        throw std::runtime_error("Number of row names must match matrix rows");
    }

    std::vector<size_t> r_index_obj = a_in_b_index(target_genes, row_names);
    std::vector<size_t> r_index = int_sort(r_index_obj, false);
    // std::vector<int> r_index = sorted_index_obj.cast<std::vector<int>>();

    // Create output matrix
    std::vector<double> sub_data(r_index.size() * num_time_points);
    std::vector<std::string> sub_row_names;

    // Copy selected rows
    for (size_t i = 0; i < r_index.size(); ++i) {
        int src_row = r_index[i];
        for (size_t j = 0; j < num_time_points; ++j) {
            sub_data[i * num_time_points + j] = data[src_row * num_time_points + j];
        }
        sub_row_names.push_back(row_names[src_row]);
    }

    // Create numpy array
    py::array_t<double> result({static_cast<py::ssize_t>(r_index.size()), 
                              static_cast<py::ssize_t>(num_time_points)},
                             sub_data.data());

    // Return both the matrix and row names
    // get colnames from result by using the num_time_points
    std::vector<std::string> col_names;
    for (size_t i = 0; i < num_time_points; ++i) {
        col_names.push_back(std::to_string(i + 1));
    }
   

    FBNMatrix output = FBNMatrix(result, sub_row_names, col_names);
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
    if (!main_parameters.contains("currentStates") || 
        !main_parameters.contains("previousStates") ||
        !main_parameters.contains("currentStates_c") || 
        !main_parameters.contains("previousStates_c")) {
        throw std::runtime_error("Missing required parameters in main_parameters");
    }

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

    // Import required functions
    // py::module fbn_utils = py::module::import("fbn_utils");
    // auto extract_gene_states = fbn_utils.attr("extract_gene_states");
    // auto mrbind = fbn_utils.attr("mrbind");
    // debug_str("----------------------current_states----------------------");
    // debug_str(py::str(get_current_states));
    // debug_str("----------------------previous_states----------------------");
    // debug_str(py::str(get_previous_states));
    // debug_str("----------------------current_states_c----------------------");
    // debug_str(py::str(get_current_states_c));
    // debug_str("----------------------previous_states_c----------------------");
    // debug_str(py::str(get_previous_states_c));
    // debug_str("----------------------rownames----------------------");
    // debug_str(py::str(gget_row_names));

    py::list result;
    for (int i = 0; i < temporal; i++) {
        py::array_t<double> current_state = get_current_states[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state = get_previous_states[i].cast<py::array_t<double>>();
        py::array_t<double> current_state_c = get_current_states_c[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state_c = get_previous_states_c[i].cast<py::array_t<double>>();

        // Check array dimensions
        if (previous_state.ndim() != 2 || previous_state.shape(1) < temporal + 2) {
            throw std::runtime_error("Not enough states for this temporal");
        }

        int n_state = previous_state.shape(1) - 1;
        int start_col = i + 1;
        int end_col = n_state + 1;

        // Create slices with bounds checking
        if (start_col >= end_col) {
            throw std::runtime_error("Invalid column range for slicing");
        }

        // Extract submatrices with proper bounds checking
        auto t_current_state = current_state.attr("__getitem__")(
            py::make_tuple(py::ellipsis(), py::slice(start_col, end_col, 1))).cast<py::array_t<double>>();
        
        auto t_previous_state = previous_state.attr("__getitem__")(
            py::make_tuple(py::ellipsis(), py::slice(0, n_state - i, 1))).cast<py::array_t<double>>();
        
        auto t_current_state_c = current_state_c.attr("__getitem__")(
            py::make_tuple(py::ellipsis(), py::slice(start_col, end_col, 1))).cast<py::array_t<double>>();
        
        auto t_previous_state_c = previous_state_c.attr("__getitem__")(
            py::make_tuple(py::ellipsis(), py::slice(0, n_state - i, 1))).cast<py::array_t<double>>();

        // Extract gene states with validation
        FBNMatrix extracted_condition = extract_gene_states(t_previous_state, conditional_genes, get_row_names);
        FBNMatrix extracted_target = extract_gene_states(t_current_state, target_gene, get_row_names);
        FBNMatrix extracted_condition_c = extract_gene_states(t_previous_state_c, target_gene, get_row_names);
        FBNMatrix extracted_target_c = extract_gene_states(t_current_state_c, conditional_genes, get_row_names);

        // debug_function(extracted_condition.matrix_t());
        // debug_function(extracted_target.matrix_t());
        // debug_function(extracted_condition_c.matrix_t());
        // debug_function(extracted_target_c.matrix_t());
        // Combine matrices
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
}