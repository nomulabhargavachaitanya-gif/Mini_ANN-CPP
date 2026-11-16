#include "Layer.hpp"

#include <stdexcept>
#include <utility>

Layer::Layer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> act)
    : num_inputs(numInputsPerNeuron), activation(std::move(act)) {
    if (numNeurons <= 0) {
        throw std::invalid_argument("Layer numNeurons must be positive.");
    }
    if (numInputsPerNeuron <= 0) {
        throw std::invalid_argument("Layer numInputsPerNeuron must be positive.");
    }
    if (!activation) {
        throw std::invalid_argument("Layer activation cannot be null.");
    }

    neurons.reserve(static_cast<std::size_t>(numNeurons));
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron, activation, numNeurons);
    }
}

MathVector Layer::evaluate(const MathVector& inputs) const {
    std::vector<double> outputs(neurons.size());
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].evaluate(inputs);
    }
    return MathVector(outputs);
}

MathVector Layer::forward(const MathVector& inputs) {
    std::vector<double> outputs(neurons.size());
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].forward(inputs);
    }
    return MathVector(outputs);
}

std::size_t Layer::numNeurons() const noexcept { return neurons.size(); }
int Layer::numInputs() const noexcept { return num_inputs; }
std::string Layer::activationName() const { return activation->name(); }

const std::vector<Neuron>& Layer::getNeurons() const noexcept { return neurons; }
std::vector<Neuron>& Layer::getNeurons() noexcept { return neurons; }
