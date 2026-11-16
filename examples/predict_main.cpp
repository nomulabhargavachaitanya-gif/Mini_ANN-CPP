#include <iomanip>
#include <iostream>
#include <vector>

#include "NeuralNetwork.hpp"

int main() {
    NeuralNetwork myNet;

    std::cout << "Loading trained model architecture and weights...\n";
    myNet.loadModel("data/trained_classification_model.csv");

    const std::vector<double> rawInput = {0.8, 0.7, 0.6, 0.5};
    const MathVector testInput(rawInput);

    std::cout << "\nRunning prediction...\n";
    const MathVector prediction = myNet.predict(testInput);

    std::cout << "Input Data: (" << rawInput[0] << ", " << rawInput[1] << ", "
              << rawInput[2] << ", " << rawInput[3] << ")\n";
    std::cout << "Network Output (Probability): " << std::fixed << std::setprecision(4)
              << prediction[0] << "\n";

    const int predictedClass = (prediction[0] >= 0.5) ? 1 : 0;
    std::cout << "Predicted Class: " << predictedClass << "\n";

    return 0;
}
