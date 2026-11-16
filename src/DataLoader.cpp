#include "DataLoader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

bool DataLoader::loadCSV(const std::string& filename,
                         std::vector<std::vector<double>>& X,
                         std::vector<std::vector<double>>& y,
                         int numTargets) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }

        if (row.size() > static_cast<std::size_t>(numTargets)) {
            std::vector<double> featureRow(row.begin(), row.end() - numTargets);
            std::vector<double> targetRow(row.end() - numTargets, row.end());

            X.push_back(featureRow);
            y.push_back(targetRow);
        }
    }

    return true;
}
