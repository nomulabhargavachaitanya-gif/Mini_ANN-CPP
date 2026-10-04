#include "DataLoader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

bool DataLoader::loadCSV(const string& filename, 
                         vector<vector<double>>& X, 
                         vector<vector<double>>& y, 
                         int numTargets) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }

        
        if (row.size() > numTargets) {
            vector<double> feature_row(row.begin(), row.end() - numTargets);
            vector<double> target_row(row.end() - numTargets, row.end());
            
            X.push_back(feature_row);
            y.push_back(target_row);
        }
    }

    file.close();
    return true;
}