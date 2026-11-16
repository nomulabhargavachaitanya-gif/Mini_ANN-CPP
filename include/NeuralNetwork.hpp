#pragma once

#include <memory>
#include <string>
#include <vector>
#include "Activations.hpp"
#include "Layer.hpp"
#include "LossFunction.hpp"
#include "Optimizer.hpp"

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    std::unique_ptr<IOptimizer> optimizer;
    std::unique_ptr<ILossFunction> lossFunction;

public:
    NeuralNetwork(std::unique_ptr<IOptimizer> opt, std::unique_ptr<ILossFunction> loss);
    NeuralNetwork(IOptimizer* opt, ILossFunction* loss);
    ~NeuralNetwork() = default;

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = default;
    NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;

    void addLayer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation);
    void addLayer(int numNeurons, int numInputsPerNeuron, IActivation* activation);

    MathVector predict(const MathVector& inputs);

    void train(const std::vector<MathVector>& X, const std::vector<MathVector>& y, int epochs);
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs);

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);
};
