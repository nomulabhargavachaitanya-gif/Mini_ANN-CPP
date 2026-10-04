#include "NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

NeuralNetwork::NeuralNetwork(IOptimizer* opt, ILossFunction* loss) {
    this->optimizer = opt;
    this->lossFunction = loss;
}

NeuralNetwork::~NeuralNetwork() {
    delete optimizer;
    delete lossFunction;
}

void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron, IActivation* activation) {
    layers.push_back(Layer(numNeurons, numInputsPerNeuron, activation));
}

MathVector NeuralNetwork::predict(const MathVector& inputs) {
    MathVector current_signal = inputs;
    for (size_t i = 0; i < layers.size(); ++i) {
        current_signal = layers[i].forward(current_signal);
    }
    return current_signal;
}

void NeuralNetwork::train(const vector<MathVector>& X, const vector<MathVector>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t example = 0; example < X.size(); ++example) {
            
            MathVector prediction = predict(X[example]);
            for(size_t i = 0; i < prediction.data.size(); ++i) {
                total_loss += lossFunction->calculate(prediction.data[i], y[example].data[i]);
            }

            Layer& outputLayer = layers.back();
            for (size_t i = 0; i < outputLayer.neurons.size(); ++i) {
                Neuron& n = outputLayer.neurons[i];
                double dL_da = lossFunction->derivative(n.last_a, y[example].data[i]);
                double da_dz = n.activationFunction->derivative(n.last_z);
                n.delta = dL_da * da_dz;
            }

            for (int l = layers.size() - 2; l >= 0; --l) {
                Layer& hiddenLayer = layers[l];
                Layer& nextLayer = layers[l + 1];

                for (size_t i = 0; i < hiddenLayer.neurons.size(); ++i) {
                    Neuron& hidden_n = hiddenLayer.neurons[i];
                    double error_sum = 0.0;
                    
                    for (size_t j = 0; j < nextLayer.neurons.size(); ++j) {
                        Neuron& next_n = nextLayer.neurons[j];
                        error_sum += next_n.delta * next_n.weights.data[i];
                    }
                    
                    double da_dz = hidden_n.activationFunction->derivative(hidden_n.last_z);
                    hidden_n.delta = error_sum * da_dz;
                }
            }

            for (size_t l = 0; l < layers.size(); ++l) {
                Layer& layer = layers[l];
                for (size_t i = 0; i < layer.neurons.size(); ++i) {
                    Neuron& n = layer.neurons[i];
                    
                    MathVector gradients = n.last_inputs * n.delta;
                    MathVector weight_update = optimizer->calculateUpdate(gradients, n.velocity);
                    double bias_update = optimizer->calculateUpdate(n.delta, n.bias_velocity);
                    
                    n.weights = n.weights - weight_update;
                    n.bias = n.bias - bias_update;
                }
            }
        }
        
        if (epoch % 500 == 0) cout << "Epoch " << epoch << " Loss: " << total_loss / X.size() << "\n";
    }
}

void NeuralNetwork::train(const vector<vector<double>>& X_raw, const vector<vector<double>>& y_raw, int epochs) {
    vector<MathVector> X;
    vector<MathVector> y;

    for (size_t i = 0; i < X_raw.size(); ++i) {
        X.push_back(MathVector(X_raw[i]));
    }
    for (size_t i = 0; i < y_raw.size(); ++i) {
        y.push_back(MathVector(y_raw[i]));
    }

    this->train(X, y, epochs); 
}

void NeuralNetwork::saveModel(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) return;
    
    if (layers.empty()) return;
    file << layers[0].neurons[0].weights.data.size();
    for (const auto& layer : layers) file << "," << layer.neurons.size();
    file << "\n";

    for (size_t l = 0; l < layers.size(); ++l) {
        for (size_t n = 0; n < layers[l].neurons.size(); ++n) {
            const Neuron& neuron = layers[l].neurons[n];
            file << l << "," << n << "," << neuron.bias;
            for (double w : neuron.weights.data) file << "," << w;
            file << "\n";
        }
    }
    file.close();
    cout << "Model saved to " << filename << "\n";
}

void NeuralNetwork::loadModel(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) return;
    
    string line;
    getline(file, line); 
    
    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        
        getline(ss, val, ','); int l = stoi(val);
        getline(ss, val, ','); int n = stoi(val);
        getline(ss, val, ','); layers[l].neurons[n].bias = stod(val);
        
        int w_idx = 0;
        while (getline(ss, val, ',')) {
            layers[l].neurons[n].weights.data[w_idx++] = stod(val);
        }
    }
    file.close();
    cout << "Model loaded from " << filename << "\n";
}
