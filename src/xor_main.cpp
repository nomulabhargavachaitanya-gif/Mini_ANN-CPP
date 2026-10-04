#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main() {
    vector<vector<double>> X_train = {
        {0.0, 0.0}, 
        {0.0, 1.0},
        {1.0, 0.0}, 
        {1.0, 1.0}
    };

    vector<vector<double>> y_train = {
        {0.0}, 
        {1.0},
        {1.0}, 
        {0.0}
    };

    NeuralNetwork myNet(new Momentum(0.1, 0.9), new MSE());
    myNet.addLayer(4, 2, new Tanh());     
    myNet.addLayer(1, 4, new Sigmoid());  

    cout << "--- Training XOR ---\n";
    myNet.train(X_train, y_train, 2000);

    cout << "\n--- Final Predictions ---\n";
    for (const auto& x : X_train) {
        MathVector pred = myNet.predict(MathVector(x)); 
        cout << "Input: (" << x[0] << ", " << x[1] << ") -> Predicted: " << pred.data[0] << "\n";
    }

    return 0;
}
