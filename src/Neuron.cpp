#include "Neuron.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

std::mt19937& Neuron::rng() {
    static std::mt19937 generator(std::random_device{}());
    return generator;
}

void Neuron::setSeed(unsigned int seed) {
    rng().seed(seed);
}

Neuron::Neuron(int numInputs, std::shared_ptr<IActivation> activation, int numOutputs)
    : weights(static_cast<std::size_t>(numInputs > 0 ? numInputs : 0), 0.0),
      bias(0.0),
      velocity(static_cast<std::size_t>(numInputs > 0 ? numInputs : 0), 0.0),
      bias_velocity(0.0),
      activationFunction(std::move(activation)),
      last_inputs(static_cast<std::size_t>(numInputs > 0 ? numInputs : 0), 0.0),
      last_z(0.0),
      last_a(0.0),
      delta(0.0),
      weight_gradients(static_cast<std::size_t>(numInputs > 0 ? numInputs : 0), 0.0),
      bias_gradient(0.0) {
    if (numInputs <= 0) {
        throw std::invalid_argument("Neuron numInputs must be positive.");
    }
    if (!activationFunction) {
        throw std::invalid_argument("Neuron activationFunction cannot be null.");
    }

    const int fanOut = (numOutputs > 0) ? numOutputs : 1;
    if (activationFunction->name() == "ReLU") {
        // He normal initialization for ReLU
        const double stddev = std::sqrt(2.0 / static_cast<double>(numInputs));
        std::normal_distribution<double> dist(0.0, stddev);
        for (int i = 0; i < numInputs; ++i) {
            weights.data[static_cast<std::size_t>(i)] = dist(rng());
        }
    } else {
        // Xavier / Glorot uniform initialization for Sigmoid and Tanh
        const double limit = std::sqrt(6.0 / static_cast<double>(numInputs + fanOut));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (int i = 0; i < numInputs; ++i) {
            weights.data[static_cast<std::size_t>(i)] = dist(rng());
        }
    }
}

double Neuron::evaluate(const MathVector& inputs) const {
    const double z = (weights * inputs) + bias;
    return activationFunction->activate(z);
}

double Neuron::forward(const MathVector& inputs) {
    last_inputs = inputs;
    last_z = (weights * inputs) + bias;
    last_a = activationFunction->activate(last_z);
    return last_a;
}

const MathVector& Neuron::getWeights() const noexcept { return weights; }
MathVector& Neuron::getWeights() noexcept { return weights; }
double Neuron::getBias() const noexcept { return bias; }
double& Neuron::getBias() noexcept { return bias; }

const MathVector& Neuron::getWeightGradients() const noexcept { return weight_gradients; }
double Neuron::getBiasGradient() const noexcept { return bias_gradient; }

const MathVector& Neuron::getLastInputs() const noexcept { return last_inputs; }
double Neuron::getLastZ() const noexcept { return last_z; }
double Neuron::getLastA() const noexcept { return last_a; }
