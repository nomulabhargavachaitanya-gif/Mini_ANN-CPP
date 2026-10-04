#pragma once
#include "MathVector.hpp"
#include "activations.hpp"
#include <vector>

using namespace std;

class Neuron {
    friend class NeuralNetwork; 
    friend class Layer;
private:
    MathVector weights;
    double bias;
    MathVector velocity; 
    double bias_velocity;
    IActivation* activationFunction;
    
    MathVector last_inputs;
    double last_z;
    double last_a;
    double delta;

public:
    Neuron(int numInputs, IActivation* activation);
    double forward(const MathVector& inputs);
};
