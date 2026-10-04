#pragma once
#include <cmath>
#include "MathVector.hpp"

using namespace std;

class IActivation {
public:
    virtual double activate(double x) = 0;
    virtual double derivative(double x) = 0;
    virtual ~IActivation() = default;
};

class Sigmoid : public IActivation {
public:
    double activate(double x) override { return 1.0 / (1.0 + exp(-x)); }
    double derivative(double x) override { double sig = activate(x); return sig * (1.0 - sig); }
};

class ReLU : public IActivation {
public:
    double activate(double x) override { return (x > 0.0) ? x : 0.0; }
    double derivative(double x) override { return (x > 0.0) ? 1.0 : 0.0; }
};

class Tanh : public IActivation {
public:
    double activate(double x) override { return tanh(x); }
    double derivative(double x) override { double t = tanh(x); return 1.0 - (t * t); }
};