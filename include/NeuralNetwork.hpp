#pragma once
#include "Layer.hpp"
#include "activations.hpp"
#include "lossfunction.hpp"
#include "optimiser.hpp"
#include <vector>
#include <string>

using namespace std;

class NeuralNetwork {
private:
    vector<Layer> layers;
    IOptimizer* optimizer;
    ILossFunction* lossFunction;

public:
    NeuralNetwork(IOptimizer* opt, ILossFunction* loss);
    ~NeuralNetwork();

    void addLayer(int numNeurons, int numInputsPerNeuron, IActivation* activation);
    MathVector predict(const MathVector& inputs);
    
    void train(const vector<MathVector>& X, const vector<MathVector>& y, int epochs);
    void train(const vector<vector<double>>& X, const vector<vector<double>>& y, int epochs);
    
    void saveModel(const string& filename);
    void loadModel(const string& filename);
};
