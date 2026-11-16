#include "Activations.hpp"

#include <cmath>
#include <stdexcept>

double Sigmoid::activate(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::derivative(double x) const {
    const double sig = activate(x);
    return sig * (1.0 - sig);
}

std::string Sigmoid::name() const {
    return "Sigmoid";
}

double ReLU::activate(double x) const {
    return (x > 0.0) ? x : 0.0;
}

double ReLU::derivative(double x) const {
    return (x > 0.0) ? 1.0 : 0.0;
}

std::string ReLU::name() const {
    return "ReLU";
}

double Tanh::activate(double x) const {
    return std::tanh(x);
}

double Tanh::derivative(double x) const {
    const double t = std::tanh(x);
    return 1.0 - (t * t);
}

std::string Tanh::name() const {
    return "Tanh";
}

std::shared_ptr<IActivation> createActivation(const std::string& typeName) {
    if (typeName == "Sigmoid") {
        return std::make_shared<Sigmoid>();
    }
    if (typeName == "ReLU") {
        return std::make_shared<ReLU>();
    }
    if (typeName == "Tanh") {
        return std::make_shared<Tanh>();
    }
    throw std::invalid_argument("Unknown activation function: " + typeName);
}
