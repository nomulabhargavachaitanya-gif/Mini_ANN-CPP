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
    if (!DataLoader::loadCSV(dataset, X_train, y_train, targetColumns)) {
        return 1;
    }

    const int numFeatures = static_cast<int>(X_train[0].size());
    const int numTargets = static_cast<int>(y_train[0].size());

    std::cout << "Successfully loaded " << X_train.size() << " rows.\n";
    std::cout << "Features per input: " << numFeatures
              << " | Targets per output: " << numTargets << "\n";

    NeuralNetwork myNet(
        std::make_unique<Momentum>(0.01, 0.9),
        std::make_unique<BinaryCrossEntropy>()
    );

    myNet.addLayer(8, numFeatures, std::make_shared<Tanh>());
    myNet.addLayer(4, 8, std::make_shared<Tanh>());
    myNet.addLayer(numTargets, 4, std::make_shared<Sigmoid>());

    std::cout << "\nTraining Started...\n";
    myNet.train(X_train, y_train, 1001);

    std::cout << "\n--- Quick Verification ---\n";
    const MathVector testInput(X_train[0]);
    const MathVector pred = myNet.predict(testInput);

    std::cout << "Expected Output: " << y_train[0][0] << "\n";
    std::cout << "Network Output:  " << std::fixed << std::setprecision(4) << pred[0] << "\n";

    myNet.saveModel("data/trained_classification_model.csv");

    return 0;
}
