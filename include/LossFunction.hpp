#pragma once

class ILossFunction {
public:
    virtual ~ILossFunction() = default;
    virtual double calculate(double prediction, double target) const = 0;
    virtual double derivative(double prediction, double target) const = 0;
};

class MSE : public ILossFunction {
public:
    double calculate(double prediction, double target) const override;
    double derivative(double prediction, double target) const override;
};

class BinaryCrossEntropy : public ILossFunction {
private:
    static constexpr double kEpsilon = 1e-15;

public:
    double calculate(double prediction, double target) const override;
    double derivative(double prediction, double target) const override;
};
