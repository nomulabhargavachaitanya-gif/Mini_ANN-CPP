#pragma once

#include <memory>
#include <random>
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

    MathVector weight_gradients;
    double bias_gradient;

public:
    Neuron(int numInputs, std::shared_ptr<IActivation> activation, int numOutputs = 1);

    // Pure const inference (does not mutate training cache)
    double evaluate(const MathVector& inputs) const;

    // Training forward pass (caches last_inputs, last_z, last_a for backprop)
    double forward(const MathVector& inputs);

    // Deterministic RNG control for weight initialization and training shuffle
    static void setSeed(unsigned int seed);
    static std::mt19937& rng();

    // Inspection & testing accessors
    const MathVector& getWeights() const noexcept;
    MathVector& getWeights() noexcept;
    double getBias() const noexcept;
    double& getBias() noexcept;

    const MathVector& getWeightGradients() const noexcept;
    double getBiasGradient() const noexcept;

    const MathVector& getLastInputs() const noexcept;
    double getLastZ() const noexcept;
    double getLastA() const noexcept;
};
