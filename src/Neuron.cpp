#include "Neuron.hpp"
#include <random>

using namespace std;

Neuron::Neuron(int numInputs, IActivation* activation) 
    : weights(numInputs, 0.0), velocity(numInputs, 0.0), last_inputs(numInputs, 0.0) {
    
    this->activationFunction = activation;
    this->bias = 0.0;
    this->bias_velocity = 0.0;
    this->delta = 0.0;
    this->last_z = 0.0;
    this->last_a = 0.0;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int i = 0; i < numInputs; ++i) {
        weights.data[i] = dis(gen);
    }
}

double Neuron::forward(const MathVector& inputs) {
    this->last_inputs = inputs;
    this->last_z = (this->weights * inputs) + this->bias;
    this->last_a = this->activationFunction->activate(this->last_z);
    return this->last_a;
}
