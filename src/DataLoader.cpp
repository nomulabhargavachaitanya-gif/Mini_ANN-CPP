#include "DataLoader.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

std::string trim(const std::string& str) {
    std::size_t first = 0;
    while (first < str.size() && std::isspace(static_cast<unsigned char>(str[first]))) {
        ++first;
    }
    std::size_t last = str.size();
    while (last > first && std::isspace(static_cast<unsigned char>(str[last - 1]))) {
        --last;
    }
    return str.substr(first, last - first);
}

bool tryParseDouble(const std::string& cell, double& outValue) {
    const std::string trimmed = trim(cell);
    if (trimmed.empty()) {
        return false;
    }
    try {
        std::size_t pos = 0;
        const double val = std::stod(trimmed, &pos);
        if (pos != trimmed.size()) {
            return false;
        }
        outValue = val;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

bool DataLoader::loadCSV(const std::string& filename,
                         std::vector<std::vector<double>>& X,
                         std::vector<std::vector<double>>& y,
                         int numTargets) {
    if (numTargets <= 0) {
        std::cerr << "Error: numTargets must be positive.\n";
        return false;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return false;
    }

    X.clear();
    y.clear();

    std::string line;
    bool isFirstNonEmptyLine = true;
    std::size_t expectedColumns = 0;
    std::size_t lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;
        if (trim(line).empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        bool rowValid = true;

        while (std::getline(ss, cell, ',')) {
            double val = 0.0;
            if (!tryParseDouble(cell, val)) {
                rowValid = false;
                break;
            }
            row.push_back(val);
        }

        if (!rowValid) {
            if (isFirstNonEmptyLine) {
                // Skip header line
                isFirstNonEmptyLine = false;
                continue;
            }
            std::cerr << "Error: Invalid numeric value at line " << lineNumber
                      << " in " << filename << "\n";
            X.clear();
            y.clear();
            return false;
        }

        isFirstNonEmptyLine = false;

        if (row.size() <= static_cast<std::size_t>(numTargets)) {
            std::cerr << "Error: Row at line " << lineNumber
                      << " has too few columns for " << numTargets << " target(s).\n";
            X.clear();
            y.clear();
            return false;
        }

        if (expectedColumns == 0) {
            expectedColumns = row.size();
        } else if (row.size() != expectedColumns) {
            std::cerr << "Error: Inconsistent column count at line " << lineNumber
                      << " in " << filename << "\n";
            X.clear();
            y.clear();
            return false;
        }

        const auto targetOffset = row.end() - numTargets;
        X.emplace_back(row.begin(), targetOffset);
        y.emplace_back(targetOffset, row.end());
    }

    if (X.empty()) {
        std::cerr << "Error: No valid data rows found in " << filename << "\n";
        return false;
    }

    return true;
}
