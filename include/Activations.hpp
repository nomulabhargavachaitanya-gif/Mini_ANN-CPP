#pragma once

#include <cmath>

class IActivation {
public:
    virtual ~IActivation() = default;
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
};

class Sigmoid : public IActivation {
public:
    double activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double x) const override {
        const double sig = activate(x);
        return sig * (1.0 - sig);
    }
};

class ReLU : public IActivation {
public:
    double activate(double x) const override {
        return (x > 0.0) ? x : 0.0;
    }

    double derivative(double x) const override {
        return (x > 0.0) ? 1.0 : 0.0;
    }
};

class Tanh : public IActivation {
public:
    double activate(double x) const override {
        return std::tanh(x);
    }

    double derivative(double x) const override {
        const double t = std::tanh(x);
        return 1.0 - (t * t);
    }
};
