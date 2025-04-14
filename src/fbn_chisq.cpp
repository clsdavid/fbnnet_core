#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <random> 


namespace py = pybind11;

double chisq_statistic(const std::vector<double>& observed, 
                       const std::vector<double>& expected) {
    if (observed.size() != expected.size()) {
        throw std::invalid_argument("Input vectors must have the same length");
    }

    double result = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (expected[i] == 0.0) {
            throw std::invalid_argument("Expected values cannot be zero");
        }
        double diff = observed[i] - expected[i];
        result += (diff * diff) / expected[i];
    }
    return result;
}

double chisq_statistic_np(py::array_t<double> observed, 
                         py::array_t<double> expected) {
    py::buffer_info obs_buf = observed.request();
    py::buffer_info exp_buf = expected.request();
    
    if (obs_buf.size != exp_buf.size) {
        throw std::runtime_error("Input shapes must match");
    }
    
    double* obs_ptr = static_cast<double*>(obs_buf.ptr);
    double* exp_ptr = static_cast<double*>(exp_buf.ptr);
    
    double result = 0.0;
    for (size_t i = 0; i < obs_buf.size; ++i) {
        if (exp_ptr[i] == 0.0) {
            throw std::runtime_error("Expected values cannot be zero");
        }
        double diff = obs_ptr[i] - exp_ptr[i];
        result += (diff * diff) / exp_ptr[i];
    }
    return result;
}


std::pair<double, std::vector<double>> uniform_chisq(const std::vector<double>& observed) {
    double total = std::accumulate(observed.begin(), observed.end(), 0.0);
    double expected_value = total / observed.size();
    
    std::vector<double> expected(observed.size(), expected_value);
    double chi_val = chisq_statistic(observed, expected);
    
    return {chi_val, expected};
}

std::pair<double, std::vector<double>> poisson_chisq(const std::vector<int>& counts, double lambda) {
    std::vector<double> expected(counts.size());
    double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    
    std::poisson_distribution<> poisson(lambda);
    for (size_t i = 0; i < counts.size(); ++i) {
        expected[i] = total * std::exp(-lambda) * std::pow(lambda, i) / std::tgamma(i + 1);
    }
    
    std::vector<double> obs_d(counts.begin(), counts.end());
    double chi_val = chisq_statistic(obs_d, expected);
    
    return {chi_val, expected};
}

std::pair<double, py::array_t<double>> contingency_chisq(py::array_t<int> table) {
    py::buffer_info buf = table.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be 2");
    
    int* data = static_cast<int*>(buf.ptr);
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    // Calculate row and column totals
    std::vector<int> row_totals(rows, 0);
    std::vector<int> col_totals(cols, 0);
    int grand_total = 0;
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            int val = data[i * cols + j];
            row_totals[i] += val;
            col_totals[j] += val;
            grand_total += val;
        }
    }
    
    // Calculate expected values
    py::array_t<double> expected({rows, cols});
    py::buffer_info exp_buf = expected.request();
    double* exp_data = static_cast<double*>(exp_buf.ptr);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            exp_data[i * cols + j] = (row_totals[i] * col_totals[j]) / static_cast<double>(grand_total);
        }
    }
    
    // Convert observed to double
    py::array_t<double> obs_d = table.cast<double>();
    
    double chi_val = chisq_statistic(
        py::array_t<double>(obs_d), 
        expected
    );
    
    return {chi_val, expected};
}

// Add to your module
int contingency_df(py::array_t<int> table) {
    py::buffer_info buf = table.request();
    if (buf.ndim != 2) throw std::runtime_error("Number of dimensions must be 2");
    return (buf.shape[0] - 1) * (buf.shape[1] - 1);
}

int goodness_of_fit_df(int n_categories, int n_estimated_params=0) {
    return n_categories - 1 - n_estimated_params;
}

PYBIND11_MODULE(fbn_chisq, m) {
    m.def("chisq_statistic_np", &chisq_statistic_np, 
          "Calculate chi-square statistic between observed and expected values",
          py::arg("observed"), py::arg("expected"));

    m.def("chisq_statistic", &chisq_statistic, 
        "Calculate chi-square statistic between observed and expected values",
        py::arg("observed"), py::arg("expected"));

    m.def("uniform_chisq", &uniform_chisq,
        "Chi-square test against uniform distribution",
        py::arg("observed"));

    m.def("poisson_chisq", &poisson_chisq,
        "Chi-square test against Poisson distribution",
        py::arg("counts"), py::arg("lambda"));

    m.def("contingency_chisq", &contingency_chisq,
        "Chi-square test for contingency table, Chi-Square Test of Independence",
        py::arg("table"));

    m.def("contingency_df", &contingency_df,
        "Degrees of freedom for contingency table",
        py::arg("table"));

    m.def("goodness_of_fit_df", &goodness_of_fit_df,
        "Degrees of freedom for goodness of fit test",
        py::arg("n_categories"), py::arg("n_estimated_params") = 0);
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Leshi Chen <chenleshi@hotmail.com>";
}