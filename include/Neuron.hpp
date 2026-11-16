#pragma once

#include <memory>
#include "Activations.hpp"
#include "MathVector.hpp"

class Neuron {
    friend class NeuralNetwork;
    friend class Layer;

private:
    MathVector weights;
    double bias;
    MathVector velocity;
    double bias_velocity;
    std::shared_ptr<IActivation> activationFunction;

    MathVector last_inputs;
    double last_z;
    double last_a;
    double delta;

public:
    Neuron(int numInputs, std::shared_ptr<IActivation> activation);
    double forward(const MathVector& inputs);
};
