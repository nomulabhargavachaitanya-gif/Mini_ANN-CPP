#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataLoader.hpp"
#include "NeuralNetwork.hpp"

namespace {

int g_testsPassed = 0;
int g_testsFailed = 0;

void check(bool condition, const std::string& message, int line) {
    if (condition) {
        ++g_testsPassed;
    } else {
        ++g_testsFailed;
        std::cerr << "[FAIL] Line " << line << ": " << message << "\n";
    }
}

#define CHECK(cond, msg) check((cond), (msg), __LINE__)

template <typename ExceptionType, typename Func>
bool throwsException(Func&& fn) {
    try {
        fn();
        return false;
    } catch (const ExceptionType&) {
        return true;
    } catch (...) {
        return false;
    }
}

// 1. Numerical vs. Analytical Gradient Check (MSE & BinaryCrossEntropy across multiple layers)
void testGradientCheck() {
    auto verifyNetworkGradients = [](std::unique_ptr<ILossFunction> lossFn, const std::string& label) {
        NeuralNetwork::setSeed(12345);
        NeuralNetwork net(std::make_unique<SGD>(0.01), std::move(lossFn));
        net.addLayer(4, 3, std::make_shared<Tanh>());
        net.addLayer(3, 4, std::make_shared<ReLU>());
        net.addLayer(2, 3, std::make_shared<Sigmoid>());

        const MathVector x = {0.35, -0.65, 0.85};
        const MathVector y = {1.0, 0.0};

        net.zeroGradients();
        net.accumulateGradients(x, y);

        // Save analytical gradients before finite-difference perturbations
        auto& layers = net.getLayers();
        std::vector<std::vector<MathVector>> analyticalWeightGrads(layers.size());
        std::vector<std::vector<double>> analyticalBiasGrads(layers.size());

        for (std::size_t l = 0; l < layers.size(); ++l) {
            const auto& neurons = layers[l].getNeurons();
            analyticalWeightGrads[l].resize(neurons.size());
            analyticalBiasGrads[l].resize(neurons.size());
            for (std::size_t n = 0; n < neurons.size(); ++n) {
                analyticalWeightGrads[l][n] = neurons[n].getWeightGradients();
                analyticalBiasGrads[l][n] = neurons[n].getBiasGradient();
            }
        }

        const double eps = 1e-6;
        const double tolerance = 1e-6;

        for (std::size_t l = 0; l < layers.size(); ++l) {
            auto& neurons = layers[l].getNeurons();
            for (std::size_t n = 0; n < neurons.size(); ++n) {
                // Check every weight gradient via central difference: (L(w+eps) - L(w-eps)) / (2*eps)
                auto& weights = neurons[n].getWeights();
                for (std::size_t w = 0; w < weights.size(); ++w) {
                    const double orig = weights[w];

                    weights[w] = orig + eps;
                    const double lossPlus = net.computeLoss(net.predict(x), y);

                    weights[w] = orig - eps;
                    const double lossMinus = net.computeLoss(net.predict(x), y);

                    weights[w] = orig;

                    const double numericalGrad = (lossPlus - lossMinus) / (2.0 * eps);
                    const double analyticalGrad = analyticalWeightGrads[l][n][w];
                    const double diff = std::abs(numericalGrad - analyticalGrad);
                    CHECK(diff < tolerance, label + " weight gradient check failed at L" +
                                                std::to_string(l) + "N" + std::to_string(n) +
                                                "W" + std::to_string(w));
                }

                // Check bias gradient via central difference
                double& bias = neurons[n].getBias();
                const double origBias = bias;

                bias = origBias + eps;
                const double lossPlus = net.computeLoss(net.predict(x), y);

                bias = origBias - eps;
                const double lossMinus = net.computeLoss(net.predict(x), y);

                bias = origBias;

                const double numericalBiasGrad = (lossPlus - lossMinus) / (2.0 * eps);
                const double analyticalBiasGrad = analyticalBiasGrads[l][n];
                const double biasDiff = std::abs(numericalBiasGrad - analyticalBiasGrad);
                CHECK(biasDiff < tolerance, label + " bias gradient check failed at L" +
                                                std::to_string(l) + "N" + std::to_string(n));
            }
        }
    };

    verifyNetworkGradients(std::make_unique<MSE>(), "MSE");
    verifyNetworkGradients(std::make_unique<BinaryCrossEntropy>(), "BinaryCrossEntropy");
}

// 2. Const predict() invariance test
void testConstPredictDoesNotMutateState() {
    NeuralNetwork::setSeed(99);
    NeuralNetwork net(std::make_unique<SGD>(0.01), std::make_unique<MSE>());
    net.addLayer(3, 2, std::make_shared<Tanh>());
    net.addLayer(1, 3, std::make_shared<Sigmoid>());

    const MathVector trainInput = {0.4, 0.7};
    const MathVector trainTarget = {1.0};
    net.zeroGradients();
    net.accumulateGradients(trainInput, trainTarget);

    const double cachedZBefore = net.getLayers()[0].getNeurons()[0].getLastZ();
    const double cachedABefore = net.getLayers()[0].getNeurons()[0].getLastA();

    const NeuralNetwork& constNet = net;
    const MathVector out = constNet.predict(MathVector{0.9, -0.2});
    CHECK(out.size() == 1, "predict() returns 1D output vector");

    const double cachedZAfter = net.getLayers()[0].getNeurons()[0].getLastZ();
    const double cachedAAfter = net.getLayers()[0].getNeurons()[0].getLastA();
    CHECK(cachedZBefore == cachedZAfter && cachedABefore == cachedAAfter,
          "const predict() must not mutate neuron forward caches");
}

// 3. Save/Load full precision round-trip & out-of-bounds mismatch protection
void testModelSaveLoadAndValidation() {
    const std::string tempFile = "data/test_temp_model.csv";

    NeuralNetwork::setSeed(777);
    NeuralNetwork net(std::make_unique<Momentum>(0.05, 0.9), std::make_unique<BinaryCrossEntropy>());
    net.addLayer(8, 4, std::make_shared<Tanh>());
    net.addLayer(4, 8, std::make_shared<Tanh>());
    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    const MathVector sampleInput = {0.123456789012345, -0.987654321098765, 0.555555555555555, 0.25};
    const MathVector originalPred = net.predict(sampleInput);

    net.saveModel(tempFile);

    // Load into an empty network (auto-reconstructs 4 -> 8:Tanh -> 4:Tanh -> 1:Sigmoid)
    NeuralNetwork autoLoadedNet;
    autoLoadedNet.loadModel(tempFile);
    const MathVector loadedPred = autoLoadedNet.predict(sampleInput);

    CHECK(std::abs(originalPred[0] - loadedPred[0]) < 1e-15,
          "Saved and reloaded model predictions must match to full double precision");

    // Attempt to load a 4-8-4-1 saved model into a 4-2-1 network (Bug 1 reproduction test)
    NeuralNetwork mismatchedNet(std::make_unique<SGD>(0.01), std::make_unique<MSE>());
    mismatchedNet.addLayer(2, 4, std::make_shared<Tanh>());
    mismatchedNet.addLayer(1, 2, std::make_shared<Sigmoid>());

    CHECK(throwsException<std::runtime_error>([&]() {
        mismatchedNet.loadModel(tempFile);
    }), "loadModel must reject mismatched network topology instead of writing out of bounds");

    std::remove(tempFile.c_str());
}

// 4. Input validation checks (Bug 5 reproduction tests)
void testInputValidation() {
    NeuralNetwork net(std::make_unique<SGD>(0.01), std::make_unique<MSE>());
    net.addLayer(4, 2, std::make_shared<Tanh>());

    // Mismatched layer input count (expects 4, given 3)
    CHECK(throwsException<std::invalid_argument>([&]() {
        net.addLayer(1, 3, std::make_shared<Sigmoid>());
    }), "addLayer must reject layer whose numInputs != previous layer numNeurons");

    net.addLayer(1, 4, std::make_shared<Sigmoid>());

    // Mismatched X and y row counts
    const std::vector<std::vector<double>> X = {{0.0, 1.0}, {1.0, 0.0}};
    const std::vector<std::vector<double>> yMismatch = {{1.0}};
    CHECK(throwsException<std::invalid_argument>([&]() {
        net.train(X, yMismatch, 10, 1, 0);
    }), "train must reject mismatched X and y row counts");

    // Wrong input dimension in predict()
    CHECK(throwsException<std::invalid_argument>([&]() {
        net.predict(MathVector{1.0, 2.0, 3.0});
    }), "predict must reject wrong input feature dimension");
}

// 5. DataLoader CSV header & malformed input tests (Bug 3 reproduction tests)
void testDataLoader() {
    const std::string headerCsv = "data/test_header_temp.csv";
    {
        std::ofstream out(headerCsv);
        out << "f1,f2,target\n";
        out << "0.1,0.2,0\n";
        out << "0.8,0.9,1\n";
    }

    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;
    const bool loaded = DataLoader::loadCSV(headerCsv, X, y, 1);
    CHECK(loaded && X.size() == 2 && y.size() == 2 && X[0].size() == 2,
          "DataLoader must skip CSV header rows cleanly");
    std::remove(headerCsv.c_str());

    // Empty CSV should return false
    const std::string emptyCsv = "data/test_empty_temp.csv";
    {
        std::ofstream out(emptyCsv);
        out << "col1,col2,target\n";
    }
    CHECK(!DataLoader::loadCSV(emptyCsv, X, y, 1),
          "DataLoader must return false when CSV has header only and no data rows");
    std::remove(emptyCsv.c_str());
}

// 6. Repeatable seeding & XOR convergence test
void testDeterministicSeedAndXorConvergence() {
    const std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    const std::vector<std::vector<double>> y = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    NeuralNetwork::setSeed(42);
    NeuralNetwork net1(std::make_unique<Momentum>(0.1, 0.9), std::make_unique<MSE>());
    net1.addLayer(4, 2, std::make_shared<Tanh>());
    net1.addLayer(1, 4, std::make_shared<Sigmoid>());
    net1.train(X, y, 2000, 1, 0, true);

    NeuralNetwork::setSeed(42);
    NeuralNetwork net2(std::make_unique<Momentum>(0.1, 0.9), std::make_unique<MSE>());
    net2.addLayer(4, 2, std::make_shared<Tanh>());
    net2.addLayer(1, 4, std::make_shared<Sigmoid>());
    net2.train(X, y, 2000, 1, 0, true);

    for (std::size_t i = 0; i < X.size(); ++i) {
        const double p1 = net1.predict(MathVector(X[i]))[0];
        const double p2 = net2.predict(MathVector(X[i]))[0];
        CHECK(std::abs(p1 - p2) < 1e-15, "Identical seeds must produce identical predictions");

        const int predLabel = (p1 >= 0.5) ? 1 : 0;
        const int trueLabel = static_cast<int>(y[i][0]);
        CHECK(predLabel == trueLabel, "XOR prediction must match target truth table");
    }
}

}  // namespace

int main() {
    std::cout << "Running MiniANN Unit & Gradient Verification Suite...\n";

    testGradientCheck();
    testConstPredictDoesNotMutateState();
    testModelSaveLoadAndValidation();
    testInputValidation();
    testDataLoader();
    testDeterministicSeedAndXorConvergence();

    std::cout << "Passed: " << g_testsPassed << " | Failed: " << g_testsFailed << "\n";
    return (g_testsFailed == 0) ? 0 : 1;
}
