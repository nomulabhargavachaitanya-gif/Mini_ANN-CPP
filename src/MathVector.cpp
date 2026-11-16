#include "MathVector.hpp"

#include <stdexcept>

MathVector::MathVector(const std::vector<double>& d) : data(d) {}

MathVector::MathVector(std::initializer_list<double> list) : data(list) {}

MathVector::MathVector(std::size_t size, double initial_value) : data(size, initial_value) {}

std::size_t MathVector::size() const noexcept {
    return data.size();
}

bool MathVector::empty() const noexcept {
    return data.empty();
}

double& MathVector::operator[](std::size_t index) {
    return data[index];
}

const double& MathVector::operator[](std::size_t index) const {
    return data[index];
}

double& MathVector::at(std::size_t index) {
    return data.at(index);
}

const double& MathVector::at(std::size_t index) const {
    return data.at(index);
}

double MathVector::operator*(const MathVector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("MathVector size mismatch in dot product.");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        result += data[i] * other.data[i];
    }
    return result;
}

MathVector MathVector::operator*(double scalar) const {
    std::vector<double> result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * scalar;
    }
    return MathVector(result);
}

MathVector MathVector::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("MathVector division by zero.");
    }
    std::vector<double> result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] / scalar;
    }
    return MathVector(result);
}

MathVector MathVector::operator-(const MathVector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("MathVector size mismatch in subtraction.");
    }
    std::vector<double> result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] - other.data[i];
    }
    return MathVector(result);
}

MathVector MathVector::operator+(const MathVector& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("MathVector size mismatch in addition.");
    }
    std::vector<double> result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] + other.data[i];
    }
    return MathVector(result);
}

MathVector& MathVector::operator+=(const MathVector& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("MathVector size mismatch in +=.");
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

MathVector& MathVector::operator-=(const MathVector& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("MathVector size mismatch in -=.");
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}
