#include "NeuralNetwork.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

NeuralNetwork::NeuralNetwork(std::unique_ptr<IOptimizer> opt, std::unique_ptr<ILossFunction> loss)
    : optimizer(std::move(opt)), lossFunction(std::move(loss)) {}

NeuralNetwork::NeuralNetwork(IOptimizer* opt, ILossFunction* loss)
    : optimizer(opt), lossFunction(loss) {}

void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron, std::shared_ptr<IActivation> activation) {
    layers.emplace_back(numNeurons, numInputsPerNeuron, std::move(activation));
}

void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron, IActivation* activation) {
    layers.emplace_back(numNeurons, numInputsPerNeuron, std::shared_ptr<IActivation>(activation));
}

MathVector NeuralNetwork::predict(const MathVector& inputs) {
    MathVector currentSignal = inputs;
    for (auto& layer : layers) {
        currentSignal = layer.forward(currentSignal);
    }
    return currentSignal;
}

void NeuralNetwork::train(const std::vector<MathVector>& X, const std::vector<MathVector>& y, int epochs) {
    if (X.empty() || layers.empty()) {
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (std::size_t example = 0; example < X.size(); ++example) {
            MathVector prediction = predict(X[example]);
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                totalLoss += lossFunction->calculate(prediction[i], y[example][i]);
            }

            // Output layer deltas
            Layer& outputLayer = layers.back();
            for (std::size_t i = 0; i < outputLayer.neurons.size(); ++i) {
                Neuron& n = outputLayer.neurons[i];
                const double dL_da = lossFunction->derivative(n.last_a, y[example][i]);
                const double da_dz = n.activationFunction->derivative(n.last_z);
                n.delta = dL_da * da_dz;
            }

            // Hidden layer deltas (backpropagation)
            for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
                Layer& hiddenLayer = layers[static_cast<std::size_t>(l)];
                Layer& nextLayer = layers[static_cast<std::size_t>(l + 1)];

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

            // Update weights and biases
            for (auto& layer : layers) {
                for (auto& n : layer.neurons) {
                    const MathVector gradients = n.last_inputs * n.delta;
                    const MathVector weightUpdate = optimizer->calculateUpdate(gradients, n.velocity);
                    const double biasUpdate = optimizer->calculateUpdate(n.delta, n.bias_velocity);

                    n.weights = n.weights - weightUpdate;
                    n.bias = n.bias - biasUpdate;
                }
            }
        }

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << (totalLoss / static_cast<double>(X.size())) << "\n";
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& X_raw,
                          const std::vector<std::vector<double>>& y_raw,
                          int epochs) {
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

    train(X, y, epochs);
}

void NeuralNetwork::saveModel(const std::string& filename) const {
    if (layers.empty()) {
        std::cerr << "Error: Cannot save an empty model.\n";
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for saving: " << filename << "\n";
        return;
    }

    file << layers[0].neurons[0].weights.size();
    for (const auto& layer : layers) {
        file << "," << layer.neurons.size();
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
        std::cerr << "Error: Could not open model file: " << filename << "\n";
        return;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return;
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string val;

        std::getline(ss, val, ',');
        const auto l = static_cast<std::size_t>(std::stoi(val));
        std::getline(ss, val, ',');
        const auto n = static_cast<std::size_t>(std::stoi(val));
        std::getline(ss, val, ',');
        layers[l].neurons[n].bias = std::stod(val);

        std::size_t wIdx = 0;
        while (std::getline(ss, val, ',')) {
            layers[l].neurons[n].weights[wIdx++] = std::stod(val);
        }
    }

    std::cout << "Model loaded from " << filename << "\n";
}
