#pragma once
#include <cmath>
#include "MathVector.hpp"

using namespace std;

class ILossFunction {
public:
    virtual double calculate(double prediction, double target) = 0;
    virtual double derivative(double prediction, double target) = 0;
    virtual ~ILossFunction() = default;
};

class MSE : public ILossFunction {
public:
    double calculate(double prediction, double target) override { return 0.5 * pow((prediction - target), 2); }
    double derivative(double prediction, double target) override { return (prediction - target); }
};

class BinaryCrossEntropy : public ILossFunction {
public:

    double calculate(double prediction, double target) override {
        double epsilon = 1e-15; 
        prediction = max(epsilon, min(1.0 - epsilon, prediction));
        return -(target * log(prediction) + (1.0 - target) * log(1.0 - prediction));
    }

    double derivative(double prediction, double target) override {
        double epsilon = 1e-15;
        prediction = max(epsilon, min(1.0 - epsilon, prediction));
        return -(target / prediction) + ((1.0 - target) / (1.0 - prediction));
    }
};