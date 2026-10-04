#pragma once
#include "Neuron.hpp"
#include "activations.hpp"
#include <vector>

using namespace std;

class Layer {
    friend class NeuralNetwork;
private:
    vector<Neuron> neurons;
public:
    Layer(int numNeurons, int numInputsPerNeuron, IActivation* activation);
    MathVector forward(const MathVector& inputs);
};
