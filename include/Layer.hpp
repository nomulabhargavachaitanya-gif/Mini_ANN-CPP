#pragma once

#include <memory>
#include <vector>
#include "Activations.hpp"
#include "Neuron.hpp"

class Layer {
    friend class NeuralNetwork;

private:
    std::vector<Neuron> neurons;

public:
    Layer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation);
    MathVector forward(const MathVector& inputs);
};
