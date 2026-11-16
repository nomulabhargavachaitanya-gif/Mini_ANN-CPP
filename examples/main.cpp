#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DataLoader.hpp"
#include "NeuralNetwork.hpp"

int main() {
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> y_train;

    const std::string dataset = "data/data.csv";
    const int targetColumns = 1;

    std::cout << "Loading dataset...\n";
    if (!DataLoader::loadCSV(dataset, X_train, y_train, targetColumns) || X_train.empty()) {
        return 1;
    }

    const int numFeatures = static_cast<int>(X_train[0].size());
    const int numTargets = static_cast<int>(y_train[0].size());

    std::cout << "Successfully loaded " << X_train.size() << " rows.\n";
    std::cout << "Features per input: " << numFeatures
              << " | Targets per output: " << numTargets << "\n";

    NeuralNetwork::setSeed(42);

    NeuralNetwork myNet(
        std::make_unique<Momentum>(0.05, 0.9),
        std::make_unique<BinaryCrossEntropy>()
    );

    myNet.addLayer(8, numFeatures, std::make_shared<Tanh>());
    myNet.addLayer(4, 8, std::make_shared<Tanh>());
    myNet.addLayer(numTargets, 4, std::make_shared<Sigmoid>());

    std::cout << "\nTraining Started...\n";
    myNet.train(X_train, y_train, 501, 8, 100, true);

    std::size_t correct = 0;
    for (std::size_t i = 0; i < X_train.size(); ++i) {
        const MathVector pred = myNet.predict(MathVector(X_train[i]));
        const int predictedLabel = (pred[0] >= 0.5) ? 1 : 0;
        const int actualLabel = (y_train[i][0] >= 0.5) ? 1 : 0;
        if (predictedLabel == actualLabel) {
            ++correct;
        }
    }

    std::cout << "\n--- Evaluation Summary ---\n";
    std::cout << "Training Accuracy: " << correct << "/" << X_train.size() << " ("
              << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(correct) / static_cast<double>(X_train.size()))
              << "%)\n";

    myNet.saveModel("data/trained_classification_model.csv");

    return 0;
}
