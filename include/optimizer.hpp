#pragma once
#include <cmath>
#include "MathVector.hpp"

using namespace std;

class IOptimizer {
protected:
    double learningRate;
public:
    IOptimizer(double lr) : learningRate(lr) {}
    virtual MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) = 0;
    virtual double calculateUpdate(double gradient, double& neuron_bias_velocity) = 0;
    virtual ~IOptimizer() = default;
};

class SGD : public IOptimizer {
public:
    SGD(double lr) : IOptimizer(lr) {}
    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) override { return gradients * learningRate; }
    double calculateUpdate(double gradient, double& neuron_bias_velocity) override { return gradient * learningRate; }
};

class Momentum : public IOptimizer {
private:
    double beta;
public:
    Momentum(double lr, double b = 0.9) : IOptimizer(lr), beta(b) {}
    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) override {
        neuron_velocity = (neuron_velocity * beta) + (gradients * learningRate);
        return neuron_velocity;
    }
    double calculateUpdate(double gradient, double& neuron_bias_velocity) override {
        neuron_bias_velocity = (neuron_bias_velocity * beta) + (gradient * learningRate);
        return neuron_bias_velocity;
    }
};