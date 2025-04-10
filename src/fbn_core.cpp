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

py::dict extract_gene_states(
    py::array_t<double>& state_matrix,
    const std::vector<std::string>& target_genes,
    const std::vector<std::string>& row_names) 
{
    // Get matrix dimensions and data
    auto buf = state_matrix.request();
    double* data = static_cast<double*>(buf.ptr);
    size_t num_genes = buf.shape[0];
    size_t num_samples = buf.shape[1];

    // Check dimensions match
    if (row_names.size() != num_genes) {
        throw std::runtime_error("Number of row names must match matrix rows");
    }

    std::vector<size_t> r_index_obj = a_in_b_index(target_genes, row_names);
    std::vector<size_t> r_index = int_sort(r_index_obj, false);
    // std::vector<int> r_index = sorted_index_obj.cast<std::vector<int>>();

    // Create output matrix
    std::vector<double> sub_data(r_index.size() * num_samples);
    std::vector<std::string> sub_row_names;

    // Copy selected rows
    for (size_t i = 0; i < r_index.size(); ++i) {
        int src_row = r_index[i];
        for (size_t j = 0; j < num_samples; ++j) {
            sub_data[i * num_samples + j] = data[src_row * num_samples + j];
        }
        sub_row_names.push_back(row_names[src_row]);
    }

    // Create numpy array
    py::array_t<double> result({static_cast<py::ssize_t>(r_index.size()), 
                              static_cast<py::ssize_t>(num_samples)},
                             sub_data.data());

    // Return both the matrix and row names
    py::dict output;
    output["matrix"] = result;
    output["rownames"] = py::cast(sub_row_names);

    return output;
}

py::list generate_temporal_gene_states(
    py::dict& main_parameters,
    const std::vector<std::string>& target_gene,
    const std::vector<std::string>& conditional_genes,
    int temporal)
{
    // Extract parameters from dictionary
    py::list get_current_states = main_parameters["currentStates"].cast<py::list>();
    py::list get_previous_states = main_parameters["previousStates"].cast<py::list>();
    py::list get_current_states_c = main_parameters["currentStates_c"].cast<py::list>();
    py::list get_previous_states_c = main_parameters["previousStates_c"].cast<py::list>();
    std::vector<std::string> get_row_names = main_parameters["rownames"].cast<std::vector<std::string>>();

    if (temporal < 1)
        temporal = 1;

    // Validate lengths
    int cur_len = py::len(get_current_states);
    if (cur_len < temporal || cur_len == 0) {
        throw std::runtime_error("getCurrentStates subscript out of bounds");
    }

    int pre_len = py::len(get_previous_states);
    if (pre_len < temporal || pre_len == 0) {
        throw std::runtime_error("getPreviousStates subscript out of bounds");
    }

    int cur_len_c = py::len(get_current_states_c);
    if (cur_len_c < temporal || cur_len_c == 0) {
        throw std::runtime_error("getCurrentStates_c subscript out of bounds");
    }

    int pre_len_c = py::len(get_previous_states_c);
    if (pre_len_c < temporal || pre_len_c == 0) {
        throw std::runtime_error("getPreviousStates_c subscript out of bounds");
    }

    // Import required functions
    // py::module fbn_utils = py::module::import("fbn_utils");
    // auto extract_gene_states = fbn_utils.attr("extract_gene_states");
    // auto mrbind = fbn_utils.attr("mrbind");

    py::list result;
    for (int i = 0; i < temporal; i++) {
        // Get matrices for current time step
        py::array_t<double> current_state = get_current_states[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state = get_previous_states[i].cast<py::array_t<double>>();
        py::array_t<double> current_state_c = get_current_states_c[i].cast<py::array_t<double>>();
        py::array_t<double> previous_state_c = get_previous_states_c[i].cast<py::array_t<double>>();

        // Get dimensions
        int n_state = previous_state.shape(1);
        if ((n_state - temporal) < 2) {
            throw std::runtime_error("No enough states for this temporal");
        }
        n_state = n_state - 1;

        // Create slices
        py::slice current_slice(i + 1, n_state + 1, 1);
        py::slice previous_slice(0, n_state - i, 1);

        // Extract submatrices using numpy's indexing
        auto t_current_state = py::array_t<double>(current_state[py::make_tuple(py::ellipsis(), current_slice)]);
        auto t_previous_state = py::array_t<double>(previous_state[py::make_tuple(py::ellipsis(), previous_slice)]);
        auto t_current_state_c = py::array_t<double>(current_state_c[py::make_tuple(py::ellipsis(), current_slice)]);
        auto t_previous_state_c = py::array_t<double>(previous_state_c[py::make_tuple(py::ellipsis(), previous_slice)]);

        // Extract gene states
        auto extracted_condition = extract_gene_states(t_previous_state, conditional_genes, get_row_names).cast<py::array_t<double>>();
        auto extracted_target = extract_gene_states(t_current_state, target_gene, get_row_names).cast<py::array_t<double>>();
        auto extracted_condition_c = extract_gene_states(t_previous_state_c, target_gene, get_row_names).cast<py::array_t<double>>();
        auto extracted_target_c = extract_gene_states(t_current_state_c, conditional_genes, get_row_names).cast<py::array_t<double>>();

        // Combine matrices
        auto concatenated_matrix = mrbind(extracted_condition, extracted_target).cast<py::array_t<double>>();
        auto concatenated_matrix_c = mrbind(extracted_condition_c, extracted_target_c).cast<py::array_t<double>>();

        // Create subresult dictionary
        py::dict subresult;
        subresult["computation_Matrix"] = concatenated_matrix;
        subresult["computation_Matrix_c"] = concatenated_matrix_c;
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