#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

struct MathVector {
    std::vector<double> data;

    MathVector() = default;
    explicit MathVector(const std::vector<double>& d) : data(d) {}
    explicit MathVector(std::size_t size, double initial_value = 0.0) : data(size, initial_value) {}

    std::size_t size() const noexcept { return data.size(); }

    double& operator[](std::size_t index) { return data[index]; }
    const double& operator[](std::size_t index) const { return data[index]; }

    // Dot Product
    double operator*(const MathVector& other) const {
        if (data.size() != other.data.size()) {
            throw std::invalid_argument("MathVector size mismatch in dot product.");
        }
        double result = 0.0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    // Scalar Multiplication
    MathVector operator*(double scalar) const {
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return MathVector(result);
    }

    // Element-wise Subtraction
    MathVector operator-(const MathVector& other) const {
        if (data.size() != other.data.size()) {
            throw std::invalid_argument("MathVector size mismatch in subtraction.");
        }
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] - other.data[i];
        }
        return MathVector(result);
    }

    // Element-wise Addition
    MathVector operator+(const MathVector& other) const {
        if (data.size() != other.data.size()) {
            throw std::invalid_argument("MathVector size mismatch in addition.");
        }
        std::vector<double> result(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] + other.data[i];
        }
        return MathVector(result);
    }
};
