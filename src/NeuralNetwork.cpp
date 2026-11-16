#include "NeuralNetwork.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

NeuralNetwork::NeuralNetwork(std::unique_ptr<IOptimizer> opt, std::unique_ptr<ILossFunction> loss)
    : optimizer(std::move(opt)), lossFunction(std::move(loss)) {
    if (!optimizer) {
        throw std::invalid_argument("NeuralNetwork optimizer cannot be null.");
    }
    if (!lossFunction) {
        throw std::invalid_argument("NeuralNetwork lossFunction cannot be null.");
    }
}

void NeuralNetwork::setSeed(unsigned int seed) {
    Neuron::setSeed(seed);
}

void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation) {
    if (numNeurons <= 0 || numInputsPerNeuron <= 0) {
        throw std::invalid_argument("Layer dimensions must be positive.");
    }
    if (!activation) {
        throw std::invalid_argument("Layer activation cannot be null.");
    }
    if (!layers.empty()) {
        const int expectedInputs = static_cast<int>(layers.back().neurons.size());
        if (numInputsPerNeuron != expectedInputs) {
            throw std::invalid_argument(
                "Layer input count (" + std::to_string(numInputsPerNeuron) +
                ") does not match previous layer neuron count (" + std::to_string(expectedInputs) + ")."
            );
        }
    }
    layers.emplace_back(numNeurons, numInputsPerNeuron, std::move(activation));
}

MathVector NeuralNetwork::predict(const MathVector& inputs) const {
    if (layers.empty()) {
        throw std::logic_error("Cannot call predict() on an empty network.");
    }
    if (static_cast<int>(inputs.size()) != layers.front().num_inputs) {
        throw std::invalid_argument("Input vector size does not match network input dimension.");
    }

    MathVector currentSignal = inputs;
    for (const auto& layer : layers) {
        currentSignal = layer.evaluate(currentSignal);
    }
    return currentSignal;
}

double NeuralNetwork::computeLoss(const MathVector& prediction, const MathVector& target) const {
    if (!lossFunction) {
        throw std::logic_error("Loss function is not configured.");
    }
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction and target sizes do not match.");
    }
    double loss = 0.0;
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        loss += lossFunction->calculate(prediction[i], target[i]);
    }
    return loss;
}

void NeuralNetwork::zeroGradients() {
    for (auto& layer : layers) {
        for (auto& n : layer.neurons) {
            std::fill(n.weight_gradients.data.begin(), n.weight_gradients.data.end(), 0.0);
            n.bias_gradient = 0.0;
        }
    }
}

double NeuralNetwork::accumulateGradients(const MathVector& x, const MathVector& target) {
    if (layers.empty()) {
        throw std::logic_error("Cannot compute gradients on an empty network.");
    }
    if (!lossFunction) {
        throw std::logic_error("Loss function is not configured.");
    }

    // Forward pass (caching activations)
    MathVector currentSignal = x;
    for (auto& layer : layers) {
        currentSignal = layer.forward(currentSignal);
    }

    const double sampleLoss = computeLoss(currentSignal, target);

    // Output layer deltas
    Layer& outputLayer = layers.back();
    for (std::size_t i = 0; i < outputLayer.neurons.size(); ++i) {
        Neuron& n = outputLayer.neurons[i];
        const double dL_da = lossFunction->derivative(n.last_a, target[i]);
        const double da_dz = n.activationFunction->derivative(n.last_z);
        n.delta = dL_da * da_dz;
    }

    // Hidden layer deltas
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        Layer& hiddenLayer = layers[static_cast<std::size_t>(l)];
        const Layer& nextLayer = layers[static_cast<std::size_t>(l + 1)];

        for (std::size_t i = 0; i < hiddenLayer.neurons.size(); ++i) {
            Neuron& hiddenNeuron = hiddenLayer.neurons[i];
            double errorSum = 0.0;

            for (std::size_t j = 0; j < nextLayer.neurons.size(); ++j) {
                const Neuron& nextNeuron = nextLayer.neurons[j];
                errorSum += nextNeuron.delta * nextNeuron.weights[i];
            }

            const double da_dz = hiddenNeuron.activationFunction->derivative(hiddenNeuron.last_z);
            hiddenNeuron.delta = errorSum * da_dz;
        }
    }

    // Accumulate gradients for each neuron
    for (auto& layer : layers) {
        for (auto& n : layer.neurons) {
            n.weight_gradients += (n.last_inputs * n.delta);
            n.bias_gradient += n.delta;
        }
    }

    return sampleLoss;
}

void NeuralNetwork::train(const std::vector<MathVector>& X,
                          const std::vector<MathVector>& y,
                          int epochs,
                          std::size_t batchSize,
                          int logInterval,
                          bool shuffle) {
    if (layers.empty()) {
        throw std::logic_error("Cannot train an empty network.");
    }
    if (!optimizer || !lossFunction) {
        throw std::logic_error("Optimizer and loss function must be set before training.");
    }
    if (X.empty()) {
        throw std::invalid_argument("Training dataset X cannot be empty.");
    }
    if (X.size() != y.size()) {
        throw std::invalid_argument("Training inputs X and targets y must have the same number of rows.");
    }
    if (batchSize == 0) {
        throw std::invalid_argument("Batch size must be at least 1.");
    }

    const std::size_t expectedInputs = static_cast<std::size_t>(layers.front().num_inputs);
    const std::size_t expectedOutputs = layers.back().neurons.size();
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != expectedInputs) {
            throw std::invalid_argument("Input feature count mismatch at row " + std::to_string(i));
        }
        if (y[i].size() != expectedOutputs) {
            throw std::invalid_argument("Target output count mismatch at row " + std::to_string(i));
        }
    }

    std::vector<std::size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle && X.size() > 1) {
            std::shuffle(indices.begin(), indices.end(), Neuron::rng());
        }

        double totalLoss = 0.0;

        for (std::size_t batchStart = 0; batchStart < X.size(); batchStart += batchSize) {
            const std::size_t batchEnd = std::min(batchStart + batchSize, X.size());
            const double currentBatchSize = static_cast<double>(batchEnd - batchStart);

            zeroGradients();

            for (std::size_t idx = batchStart; idx < batchEnd; ++idx) {
                const std::size_t sampleIdx = indices[idx];
                totalLoss += accumulateGradients(X[sampleIdx], y[sampleIdx]);
            }

            // Apply averaged batch gradients via optimizer
            for (auto& layer : layers) {
                for (auto& n : layer.neurons) {
                    const MathVector avgWeightGrad = n.weight_gradients / currentBatchSize;
                    const double avgBiasGrad = n.bias_gradient / currentBatchSize;

                    const MathVector weightUpdate = optimizer->calculateUpdate(avgWeightGrad, n.velocity);
                    const double biasUpdate = optimizer->calculateUpdate(avgBiasGrad, n.bias_velocity);

                    n.weights -= weightUpdate;
                    n.bias -= biasUpdate;
                }
            }
        }

        if (logInterval > 0 && (epoch % logInterval == 0)) {
            std::cout << "Epoch " << epoch << " | Loss: " << (totalLoss / static_cast<double>(X.size())) << "\n";
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X_raw,
                          const std::vector<std::vector<double>>& y_raw,
                          int epochs,
                          std::size_t batchSize,
                          int logInterval,
                          bool shuffle) {
    if (X_raw.size() != y_raw.size()) {
        throw std::invalid_argument("Training inputs X and targets y must have the same number of rows.");
    }

    std::vector<MathVector> X;
    std::vector<MathVector> y;
    X.reserve(X_raw.size());
    y.reserve(y_raw.size());

    for (const auto& row : X_raw) {
        X.emplace_back(row);
    }
    for (const auto& row : y_raw) {
        y.emplace_back(row);
    }

    train(X, y, epochs, batchSize, logInterval, shuffle);
}

void NeuralNetwork::saveModel(const std::string& filename) const {
    if (layers.empty()) {
        throw std::logic_error("Cannot save an empty model.");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for saving: " + filename);
    }

    file << std::setprecision(std::numeric_limits<double>::max_digits10);

    // Header format: inputSize,layer0Size:Activation0,layer1Size:Activation1,...
    file << layers.front().num_inputs;
    for (const auto& layer : layers) {
        file << "," << layer.neurons.size() << ":" << layer.activationName();
    }
    file << "\n";

    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (std::size_t n = 0; n < layers[l].neurons.size(); ++n) {
            const Neuron& neuron = layers[l].neurons[n];
            file << l << "," << n << "," << neuron.bias;
            for (double w : neuron.weights.data) {
                file << "," << w;
            }
            file << "\n";
        }
    }

    std::cout << "Model saved to " << filename << "\n";
}

void NeuralNetwork::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open model file: " + filename);
    }

    std::string headerLine;
    if (!std::getline(file, headerLine) || headerLine.empty()) {
        throw std::runtime_error("Model file is empty or missing header: " + filename);
    }
    if (!headerLine.empty() && headerLine.back() == '\r') {
        headerLine.pop_back();
    }

    std::stringstream headerStream(headerLine);
    std::string token;
    if (!std::getline(headerStream, token, ',')) {
        throw std::runtime_error("Malformed header in model file: " + filename);
    }

    const int fileInputSize = std::stoi(token);
    if (fileInputSize <= 0) {
        throw std::runtime_error("Invalid input dimension in model file.");
    }

    struct LayerSpec {
        int numNeurons;
        std::string activationName;
    };
    std::vector<LayerSpec> specs;

    while (std::getline(headerStream, token, ',')) {
        const std::size_t colonPos = token.find(':');
        if (colonPos != std::string::npos) {
            const int n = std::stoi(token.substr(0, colonPos));
            const std::string actName = token.substr(colonPos + 1);
            specs.push_back({n, actName});
        } else {
            const int n = std::stoi(token);
            specs.push_back({n, ""});
        }
    }

    if (specs.empty()) {
        throw std::runtime_error("Model header specifies zero layers.");
    }

    if (layers.empty()) {
        // Automatically reconstruct network topology from the saved header
        int prevInputs = fileInputSize;
        for (const auto& spec : specs) {
            if (spec.activationName.empty()) {
                throw std::runtime_error("Cannot auto-construct layer without saved activation type.");
            }
            addLayer(spec.numNeurons, prevInputs, createActivation(spec.activationName));
            prevInputs = spec.numNeurons;
        }
    } else {
        // Validate existing network topology against the saved header
        if (layers.size() != specs.size()) {
            throw std::runtime_error(
                "Model layer count mismatch: file has " + std::to_string(specs.size()) +
                " layers, network has " + std::to_string(layers.size()) + "."
            );
        }
        if (layers.front().num_inputs != fileInputSize) {
            throw std::runtime_error("Model input size mismatch.");
        }
        for (std::size_t i = 0; i < specs.size(); ++i) {
            if (static_cast<int>(layers[i].neurons.size()) != specs[i].numNeurons) {
                throw std::runtime_error(
                    "Neuron count mismatch at layer " + std::to_string(i) + ": file has " +
                    std::to_string(specs[i].numNeurons) + ", network has " +
                    std::to_string(layers[i].neurons.size()) + "."
                );
            }
            if (!specs[i].activationName.empty() && layers[i].activationName() != specs[i].activationName) {
                throw std::runtime_error(
                    "Activation mismatch at layer " + std::to_string(i) + ": file has " +
                    specs[i].activationName + ", network has " + layers[i].activationName() + "."
                );
            }
        }
    }

    std::size_t expectedNeurons = 0;
    for (const auto& layer : layers) {
        expectedNeurons += layer.neurons.size();
    }

    std::size_t loadedNeurons = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string val;

        if (!std::getline(ss, val, ',')) {
            throw std::runtime_error("Malformed neuron row in model file.");
        }
        const int lInt = std::stoi(val);
        if (!std::getline(ss, val, ',')) {
            throw std::runtime_error("Malformed neuron index in model file.");
        }
        const int nInt = std::stoi(val);
        if (!std::getline(ss, val, ',')) {
            throw std::runtime_error("Malformed bias value in model file.");
        }
        const double biasVal = std::stod(val);

        if (lInt < 0 || static_cast<std::size_t>(lInt) >= layers.size()) {
            throw std::runtime_error("Layer index out of bounds in model file.");
        }
        const auto l = static_cast<std::size_t>(lInt);

        if (nInt < 0 || static_cast<std::size_t>(nInt) >= layers[l].neurons.size()) {
            throw std::runtime_error("Neuron index out of bounds in model file.");
        }
        const auto n = static_cast<std::size_t>(nInt);

        Neuron& neuron = layers[l].neurons[n];
        neuron.bias = biasVal;

        std::vector<double> parsedWeights;
        parsedWeights.reserve(neuron.weights.size());
        while (std::getline(ss, val, ',')) {
            parsedWeights.push_back(std::stod(val));
        }

        if (parsedWeights.size() != neuron.weights.size()) {
            throw std::runtime_error(
                "Weight count mismatch for layer " + std::to_string(l) +
                ", neuron " + std::to_string(n) + "."
            );
        }

        neuron.weights.data = std::move(parsedWeights);
        ++loadedNeurons;
    }

    if (loadedNeurons != expectedNeurons) {
        throw std::runtime_error("Incomplete model file: missing neuron weight rows.");
    }

    std::cout << "Model loaded from " << filename << "\n";
}

const std::vector<Layer>& NeuralNetwork::getLayers() const noexcept { return layers; }
std::vector<Layer>& NeuralNetwork::getLayers() noexcept { return layers; }

