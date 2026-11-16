#include "Optimizer.hpp"

#include <stdexcept>

IOptimizer::IOptimizer(double lr) : learningRate(lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive.");
    }
}

SGD::SGD(double lr) : IOptimizer(lr) {}

MathVector SGD::calculateUpdate(const MathVector& gradients, MathVector& /*neuronVelocity*/) {
    return gradients * learningRate;
}

double SGD::calculateUpdate(double gradient, double& /*neuronBiasVelocity*/) {
    return gradient * learningRate;
}

Momentum::Momentum(double lr, double b) : IOptimizer(lr), beta(b) {
    if (b < 0.0 || b >= 1.0) {
        throw std::invalid_argument("Momentum beta must be in [0, 1).");
    }
}

MathVector Momentum::calculateUpdate(const MathVector& gradients, MathVector& neuronVelocity) {
    neuronVelocity = (neuronVelocity * beta) + (gradients * learningRate);
    return neuronVelocity;
}

double Momentum::calculateUpdate(double gradient, double& neuronBiasVelocity) {
    neuronBiasVelocity = (neuronBiasVelocity * beta) + (gradient * learningRate);
    return neuronBiasVelocity;
}
