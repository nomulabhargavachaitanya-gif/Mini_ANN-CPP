#include "Layer.hpp"

using namespace std;

Layer::Layer(int numNeurons, int numInputsPerNeuron, IActivation* activation) {
    for (int i = 0; i < numNeurons; ++i) {
        neurons.push_back(Neuron(numInputsPerNeuron, activation));
    }
}

MathVector Layer::forward(const MathVector& inputs) {
    vector<double> outputs(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i) {
        outputs[i] = neurons[i].forward(inputs);
    }
    return MathVector(outputs);
}
