#pragma once
#include <vector>
#include <stdexcept>

using namespace std;

struct MathVector {
    vector<double> data;

    MathVector(const vector<double>& d) : data(d) {}
    MathVector(int size, double initial_value = 0.0) : data(size, initial_value) {}

    // Dot Product
    double operator*(const MathVector& other) const {
        if (this->data.size() != other.data.size()) throw invalid_argument("Size mismatch for dot product.");
        double result = 0.0;
        for (size_t i = 0; i < data.size(); i++) result += this->data[i] * other.data[i];
        return result;
    }

    // Scalar Multiplication
    MathVector operator*(double scalar) const {
        vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) result[i] = data[i] * scalar;
        return MathVector(result);
    }

    // Element-wise Subtraction
    MathVector operator-(const MathVector& other) const {
        vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) result[i] = this->data[i] - other.data[i];
        return MathVector(result);
    }

    // Element-wise Addition
    MathVector operator+(const MathVector& other) const {
        vector<double> result(data.size());
        for (size_t i = 0; i < data.size(); i++) result[i] = this->data[i] + other.data[i];
        return MathVector(result);
    }
};
