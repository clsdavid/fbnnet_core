#include "fbn_column.h"
#include <sstream>
#include <stdexcept>

size_t FBNColumn::size() const {
    return values.size();
}

double FBNColumn::get_value(const std::string& row_name) const {
    for (size_t i = 0; i < row_names.size(); ++i) {
        if (row_names[i] == row_name) {
            return values[i];
        }
    }
    throw std::runtime_error("Row name not found: " + row_name);
}

std::string FBNColumn::to_string() const {
    std::ostringstream oss;
    for (size_t i = 0; i < row_names.size(); ++i) {
        oss << row_names[i] << ": " << values[i] << "\n";
    }
    return oss.str();
}

bool FBNColumn::is_none() const {
    return row_names.empty() || values.empty();
}

std::vector<std::string> FBNColumn::get_row_names() const {
    return row_names;
}

std::vector<double> FBNColumn::get_values() const {
    return values;
}
