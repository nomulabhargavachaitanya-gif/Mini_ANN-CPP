#include "Neuron.hpp"

#include <random>
#include <utility>

Neuron::Neuron(int numInputs, std::shared_ptr<IActivation> activation)
    : weights(static_cast<std::size_t>(numInputs), 0.0),
      bias(0.0),
      velocity(static_cast<std::size_t>(numInputs), 0.0),
      bias_velocity(0.0),
      activationFunction(std::move(activation)),
      last_inputs(static_cast<std::size_t>(numInputs), 0.0),
      last_z(0.0),
      last_a(0.0),
      delta(0.0) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < numInputs; ++i) {
        weights.data[static_cast<std::size_t>(i)] = dis(gen);
    }
}

double Neuron::forward(const MathVector& inputs) {
    last_inputs = inputs;
    last_z = (weights * inputs) + bias;
    last_a = activationFunction->activate(last_z);
    return last_a;
}
