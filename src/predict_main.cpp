#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

using namespace std;

int main() {
    NeuralNetwork myNet(new Momentum(0.01, 0.9), new BinaryCrossEntropy());

    myNet.addLayer(8, 4, new Tanh());     
    myNet.addLayer(4, 8, new Tanh());     
    myNet.addLayer(1, 4, new Sigmoid());  

    cout << "Loading trained weights...\n";
    myNet.loadModel("data/trained_classification_model.csv");

    vector<double> raw_input = {0.8, 0.7, 0.6, 0.5};
    MathVector test_input(raw_input);

    cout << "\nRunning prediction...\n";
    MathVector prediction = myNet.predict(test_input);
    

    cout << "Input Data: (" << raw_input[0] << ", " << raw_input[1] << ", " 
         << raw_input[2] << ", " << raw_input[3] << ")\n";    
    cout << "Network Output (Probability): " << prediction.data[0] << "\n";

    int predicted_class = (prediction.data[0] >= 0.5) ? 1 : 0;
    cout << "Predicted Class: " << predicted_class << "\n";

    return 0;
}