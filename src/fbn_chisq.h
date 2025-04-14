#ifndef FBN_CHISQ_H
#define FBN_CHISQ_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

double chisq_statistic(const std::vector<double>& observed, const std::vector<double>& expected);

double chisq_statistic_np(py::array_t<double> observed, py::array_t<double> expected);

std::pair<double, std::vector<double>> uniform_chisq(const std::vector<double>& observed);

std::pair<double, std::vector<double>> poisson_chisq(const std::vector<int>& counts, double lambda);

std::pair<double, py::array_t<double>> contingency_chisq(py::array_t<int> table);

int contingency_df(py::array_t<int> table);

int goodness_of_fit_df(int n_categories, int n_estimated_params=0);

#endif // FBN_CHISQ_H