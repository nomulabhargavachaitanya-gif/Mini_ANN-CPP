#pragma once
#include <vector>
#include <string>

using namespace std;

class DataLoader {
public:
    static bool loadCSV(const string& filename, 
                        vector<vector<double>>& X, 
                        vector<vector<double>>& y, 
                        int numTargets);
};
