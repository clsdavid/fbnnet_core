#ifndef FBN_COLUMN_H
#define FBN_COLUMN_H

#include <string>
#include <vector>
#include <pybind11/pybind11.h>

class FBNColumn {
    public:
    std::vector<std::string> row_names;
    std::vector<double> values;

    FBNColumn() = default;

    FBNColumn(const std::vector<std::string>& names,
                const std::vector<double>& vals)
        : row_names(names), values(vals) {}

    size_t size() const;
    double get_value(const std::string& row_name) const;
    std::string to_string() const;
    bool is_none() const;

    std::vector<std::string> get_row_names() const;  // NEW
    std::vector<double> get_values() const;          // NEW
};


#endif // NAMED_COLUMN_H
