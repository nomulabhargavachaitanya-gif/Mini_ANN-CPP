#include <iostream>
#include <vector>
#include <iomanip>
#include "NeuralNetwork.hpp"
#include "DataLoader.hpp"

using namespace std;

int main() {
    vector<vector<double>> X_train;
    vector<vector<double>> y_train;

    string dataset = "data/data.csv"; 
    int targetColumns = 1; 
    
    cout << "Loading dataset...\n";
    
    if (!DataLoader::loadCSV(dataset, X_train, y_train, targetColumns)) {
        return -1; 
    }

    int numFeatures = X_train[0].size();
    int numTargets = y_train[0].size();

    cout << "Successfully loaded " << X_train.size() << " rows.\n";
    cout << "Features per input: " << numFeatures << " | Targets per output: " << numTargets << "\n";

    NeuralNetwork myNet(new Momentum(0.01, 0.9), new BinaryCrossEntropy());

    myNet.addLayer(8, numFeatures, new Tanh());     
    myNet.addLayer(4, 8, new Tanh());               
    myNet.addLayer(numTargets, 4, new Sigmoid());   

    cout << "\nTraining Started...\n";
    myNet.train(X_train, y_train, 1001); 

    cout << "\n--- Quick Verification ---\n";
    MathVector test_input = MathVector(X_train[0]);
    MathVector pred = myNet.predict(test_input);
    
    cout << "Expected Output: " << y_train[0][0] << "\n";
    cout << "Network Output:  " << fixed << setprecision(4) << pred.data[0] << "\n";

    myNet.saveModel("data/trained_classification_model.csv");
    
    return 0;
}
