#include "Layer.hpp"

Layer::Layer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation) {
    neurons.reserve(static_cast<std::size_t>(numNeurons));
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron, activation);
    }
}

MathVector Layer::forward(const MathVector& inputs) {
    std::vector<double> outputs(neurons.size());
    for (std::size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].forward(inputs);
    }
    return MathVector(outputs);
}
