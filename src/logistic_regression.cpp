#include "openfhe_lab/logistic_regression.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace labml {
namespace {

using Clock = std::chrono::steady_clock;

double SecondsBetween(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

double PolynomialSigmoid(double score) {
    return 0.5 + 0.197 * score - 0.004 * score * score * score;
}

double ExactSigmoid(double score) {
    if (score >= 0.0) {
        return 1.0 / (1.0 + std::exp(-score));
    }
    const double exponential = std::exp(score);
    return exponential / (1.0 + exponential);
}

double LinearScore(const PlainModel& model, const Sample& sample) {
    if (model.weights.size() != sample.features.size()) {
        throw std::invalid_argument("Model and sample feature counts do not match");
    }
    double result = model.bias;
    for (std::size_t feature = 0; feature < model.weights.size(); ++feature) {
        result += model.weights[feature] * sample.features[feature];
    }
    return result;
}

double Accuracy(const PlainModel& model, const Dataset& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot evaluate an empty dataset");
    }
    std::size_t correct = 0;
    for (const auto& sample : data) {
        const double prediction = LinearScore(model, sample) >= 0.0 ? 1.0 : 0.0;
        if (prediction == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(data.size());
}

double ExactLogLoss(const PlainModel& model, const Dataset& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot evaluate an empty dataset");
    }
    constexpr double epsilon = 1e-12;
    double total = 0.0;
    for (const auto& sample : data) {
        const double probability = std::clamp(
            ExactSigmoid(LinearScore(model, sample)), epsilon, 1.0 - epsilon);
        total -= sample.label * std::log(probability) +
                 (1.0 - sample.label) * std::log(1.0 - probability);
    }
    return total / static_cast<double>(data.size());
}

PlaintextTrainingResult TrainPlaintext(
    const Dataset& train,
    const Dataset& test,
    std::size_t epochs,
    double learningRate) {
    if (train.empty() || test.empty()) {
        throw std::invalid_argument("Training and test datasets must not be empty");
    }
    if (epochs == 0 || learningRate <= 0.0) {
        throw std::invalid_argument("Epochs and learning rate must be positive");
    }

    PlainModel model{std::vector<double>(FeatureCount(train), 0.0), 0.0};
    PlaintextTrainingResult result;
    result.epochs.reserve(epochs);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        const auto start = Clock::now();
        std::vector<double> weightGradient(model.weights.size(), 0.0);
        double biasGradient = 0.0;
        for (const auto& sample : train) {
            const double output = PolynomialSigmoid(LinearScore(model, sample));
            const double error  = output - sample.label;
            for (std::size_t feature = 0; feature < model.weights.size(); ++feature) {
                weightGradient[feature] += sample.features[feature] * error;
            }
            biasGradient += error;
        }

        const double step = learningRate / static_cast<double>(train.size());
        for (std::size_t feature = 0; feature < model.weights.size(); ++feature) {
            model.weights[feature] -= step * weightGradient[feature];
        }
        model.bias -= step * biasGradient;
        const auto end = Clock::now();

        result.epochs.push_back(
            {epoch + 1, SecondsBetween(start, end), Accuracy(model, test), ExactLogLoss(model, train), model});
    }
    result.finalModel = model;
    return result;
}

double MaximumModelError(const PlainModel& first, const PlainModel& second) {
    if (first.weights.size() != second.weights.size()) {
        throw std::invalid_argument("Cannot compare models with different feature counts");
    }
    double maximum = std::abs(first.bias - second.bias);
    for (std::size_t feature = 0; feature < first.weights.size(); ++feature) {
        maximum = std::max(maximum, std::abs(first.weights[feature] - second.weights[feature]));
    }
    return maximum;
}

}  // namespace labml
