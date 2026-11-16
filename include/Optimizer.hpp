#pragma once

#include "MathVector.hpp"

class IOptimizer {
protected:
    double learningRate;

public:
    explicit IOptimizer(double lr);
    virtual ~IOptimizer() = default;

    virtual MathVector calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) = 0;
    virtual double calculateUpdate(double gradient, double& neuronBiasVelocity) = 0;
};

class SGD : public IOptimizer {
public:
    explicit SGD(double lr);

    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) override;
    double calculateUpdate(double gradient, double& neuronBiasVelocity) override;
};

class Momentum : public IOptimizer {
private:
    double beta;

public:
    explicit Momentum(double lr, double b = 0.9);

    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) override;
    double calculateUpdate(double gradient, double& neuronBiasVelocity) override;
};
