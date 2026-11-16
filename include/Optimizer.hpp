#pragma once

#include "MathVector.hpp"

class IOptimizer {
protected:
    double learningRate;

public:
    explicit IOptimizer(double lr) : learningRate(lr) {}
    virtual ~IOptimizer() = default;

    virtual MathVector calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) = 0;
    virtual double calculateUpdate(double gradient, double& neuronBiasVelocity) = 0;
};

class SGD : public IOptimizer {
public:
    explicit SGD(double lr) : IOptimizer(lr) {}

    MathVector calculateUpdate(const MathVector& gradients, MathVector& /*neuronVelocity*/) override {
        return gradients * learningRate;
    }

    double calculateUpdate(double gradient, double& /*neuronBiasVelocity*/) override {
        return gradient * learningRate;
    }
};

class Momentum : public IOptimizer {
private:
    double beta;

public:
    explicit Momentum(double lr, double b = 0.9) : IOptimizer(lr), beta(b) {}

    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) override {
        neuronVelocity = (neuronVelocity * beta) + (gradients * learningRate);
        return neuronVelocity;
    }

    double calculateUpdate(double gradient, double& neuronBiasVelocity) override {
        neuronBiasVelocity = (neuronBiasVelocity * beta) + (gradient * learningRate);
        return neuronBiasVelocity;
    }
};
