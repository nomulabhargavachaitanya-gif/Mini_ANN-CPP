#pragma once

#include <cstddef>
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
    NeuralNetwork() = default;
    NeuralNetwork(std::unique_ptr<IOptimizer> opt, std::unique_ptr<ILossFunction> loss);
    ~NeuralNetwork() = default;

    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = default;
    NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;

    static void setSeed(unsigned int seed);

    void addLayer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation);

    // Const inference method (does not modify network state)
    MathVector predict(const MathVector& inputs) const;

    // Loss & gradient helpers (used by train() and gradient verification tests)
    double computeLoss(const MathVector& prediction, const MathVector& target) const;
    void zeroGradients();
    double accumulateGradients(const MathVector& x, const MathVector& target);

    void train(const std::vector<MathVector>& X,
               const std::vector<MathVector>& y,
               int epochs,
               std::size_t batchSize = 1,
               int logInterval = 500,
               bool shuffle = true);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs,
               std::size_t batchSize = 1,
               int logInterval = 500,
               bool shuffle = true);

    void saveModel(const std::string& filename) const;
    void loadModel(const std::string& filename);

    const std::vector<Layer>& getLayers() const noexcept;
    std::vector<Layer>& getLayers() noexcept;
};
