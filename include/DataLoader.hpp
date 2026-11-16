#pragma once

#include <string>
#include <vector>

class DataLoader {
public:
    static bool loadCSV(const std::string& filename,
                        std::vector<std::vector<double>>& X,
                        std::vector<std::vector<double>>& y,
                        int numTargets);
};
