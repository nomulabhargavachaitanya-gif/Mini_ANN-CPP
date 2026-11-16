#pragma once

#include <algorithm>
#include <cmath>

class ILossFunction {
public:
    virtual ~ILossFunction() = default;
    virtual double calculate(double prediction, double target) const = 0;
    virtual double derivative(double prediction, double target) const = 0;
};

class MSE : public ILossFunction {
public:
    double calculate(double prediction, double target) const override {
        const double diff = prediction - target;
        return 0.5 * diff * diff;
    }

    double derivative(double prediction, double target) const override {
        return prediction - target;
    }
};

class BinaryCrossEntropy : public ILossFunction {
private:
    static constexpr double kEpsilon = 1e-15;

public:
    double calculate(double prediction, double target) const override {
        const double p = std::clamp(prediction, kEpsilon, 1.0 - kEpsilon);
        return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
    }

    double derivative(double prediction, double target) const override {
        const double p = std::clamp(prediction, kEpsilon, 1.0 - kEpsilon);
        return -(target / p) + ((1.0 - target) / (1.0 - p));
    }
};
