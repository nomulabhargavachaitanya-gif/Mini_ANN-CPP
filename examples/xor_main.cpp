#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "NeuralNetwork.hpp"

int main() {
    const std::vector<std::vector<double>> X_train = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    const std::vector<std::vector<double>> y_train = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    NeuralNetwork myNet(
        std::make_unique<Momentum>(0.1, 0.9),
        std::make_unique<MSE>()
    );
    myNet.addLayer(4, 2, std::make_shared<Tanh>());
    myNet.addLayer(1, 4, std::make_shared<Sigmoid>());

    std::cout << "--- Training XOR ---\n";
    myNet.train(X_train, y_train, 2000);

    std::cout << "\n--- Final Predictions ---\n";
    std::cout << std::fixed << std::setprecision(4);
    for (const auto& x : X_train) {
        const MathVector pred = myNet.predict(MathVector(x));
        std::cout << "Input: (" << x[0] << ", " << x[1] << ") -> Predicted: " << pred[0] << "\n";
    }

    return 0;
}
