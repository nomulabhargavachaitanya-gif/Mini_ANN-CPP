#include "LossFunction.hpp"

#include <algorithm>
#include <cmath>

double MSE::calculate(double prediction, double target) const {
    const double diff = prediction - target;
    return 0.5 * diff * diff;
}

double MSE::derivative(double prediction, double target) const {
    return prediction - target;
}

double BinaryCrossEntropy::calculate(double prediction, double target) const {
    const double p = std::clamp(prediction, kEpsilon, 1.0 - kEpsilon);
    return -(target * std::log(p) + (1.0 - target) * std::log(1.0 - p));
}

double BinaryCrossEntropy::derivative(double prediction, double target) const {
    const double p = std::clamp(prediction, kEpsilon, 1.0 - kEpsilon);
    return -(target / p) + ((1.0 - target) / (1.0 - p));
}
