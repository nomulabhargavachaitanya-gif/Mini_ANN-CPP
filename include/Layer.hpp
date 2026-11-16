#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include "Activations.hpp"
#include "Neuron.hpp"

class Layer {
    friend class NeuralNetwork;

private:
    int num_inputs;
    std::shared_ptr<IActivation> activation;
    std::vector<Neuron> neurons;

public:
    Layer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> act);

    MathVector evaluate(const MathVector& inputs) const;
    MathVector forward(const MathVector& inputs);

    std::size_t numNeurons() const noexcept;
    int numInputs() const noexcept;
    std::string activationName() const;

    const std::vector<Neuron>& getNeurons() const noexcept;
    std::vector<Neuron>& getNeurons() noexcept;
};
