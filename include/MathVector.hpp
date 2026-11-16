#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

struct MathVector {
    std::vector<double> data;

    MathVector() = default;
    explicit MathVector(const std::vector<double>& d);
    MathVector(std::initializer_list<double> list);
    explicit MathVector(std::size_t size, double initial_value = 0.0);

    std::size_t size() const noexcept;
    bool empty() const noexcept;

    double& operator[](std::size_t index);
    const double& operator[](std::size_t index) const;

    double& at(std::size_t index);
    const double& at(std::size_t index) const;

    // Dot Product
    double operator*(const MathVector& other) const;

    // Scalar Multiplication & Division
    MathVector operator*(double scalar) const;
    MathVector operator/(double scalar) const;

    // Element-wise Operations
    MathVector operator-(const MathVector& other) const;
    MathVector operator+(const MathVector& other) const;
    MathVector& operator+=(const MathVector& other);
    MathVector& operator-=(const MathVector& other);
};
