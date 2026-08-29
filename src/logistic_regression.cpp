// Nesterov theta/phi update adapted from lr_nag.cpp in
// openfheorg/openfhe-logreg-training-examples (BSD-2-Clause).
// Copyright (c) 2023, Duality Technologies Inc. All rights reserved.
// See THIRD_PARTY_NOTICES.md for the retained upstream license.
#include "openfhe_lab/logistic_regression.hpp"

#include "math/chebyshev.h"

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

const std::vector<double>& SigmoidChebyshevCoefficients() {
    static const auto coefficients = lbcrypto::EvalChebyshevCoefficients(
        [](double value) { return 1.0 / (1.0 + std::exp(-value)); },
        kSigmoidApproximationLowerBound,
        kSigmoidApproximationUpperBound,
        kSigmoidApproximationDegree);
    return coefficients;
}

}  // namespace

std::string OptimizerName(Optimizer optimizer) {
    switch (optimizer) {
        case Optimizer::GradientDescent:
            return "gd";
        case Optimizer::NesterovAcceleratedGradient:
            return "nag";
    }
    throw std::invalid_argument("Unknown optimizer");
}

void ValidateOptimizerConfiguration(const OptimizerConfiguration& configuration) {
    OptimizerName(configuration.method);
    if (!std::isfinite(configuration.momentum) || configuration.momentum < 0.0 ||
        configuration.momentum >= 1.0) {
        throw std::invalid_argument("Momentum must be finite and in [0, 1)");
    }
}

double PolynomialSigmoid(double score) {
    const auto& coefficients = SigmoidChebyshevCoefficients();
    const double normalized = -1.0 + 2.0 *
        (score - kSigmoidApproximationLowerBound) /
        (kSigmoidApproximationUpperBound - kSigmoidApproximationLowerBound);

    // Clenshaw recurrence for c[0]/2 + sum(c[k] * T_k(normalized)).
    // EvalChebyshevCoefficients and EvalLogistic use this same c[0]/2
    // convention in OpenFHE 1.1.2.
    double next = 0.0;
    double nextNext = 0.0;
    for (std::size_t index = coefficients.size() - 1; index > 0; --index) {
        const double current = 2.0 * normalized * next - nextNext + coefficients[index];
        nextNext = next;
        next = current;
    }
    return normalized * next - nextNext + coefficients.front() / 2.0;
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
    double learningRate,
    const OptimizerConfiguration& optimizer) {
    ValidateOptimizerConfiguration(optimizer);
    if (train.empty() || test.empty()) {
        throw std::invalid_argument("Training and test datasets must not be empty");
    }
    if (epochs == 0 || !std::isfinite(learningRate) || learningRate <= 0.0) {
        throw std::invalid_argument("Epochs and learning rate must be positive and finite");
    }

    const bool useMomentum =
        optimizer.method == Optimizer::NesterovAcceleratedGradient && optimizer.momentum > 0.0;
    // model is the look-ahead theta; previousStep stores the unaccelerated phi.
    PlainModel model{std::vector<double>(FeatureCount(train), 0.0), 0.0};
    PlainModel previousStep = model;
    PlaintextTrainingResult result;
    result.optimizer = optimizer;
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
            const double nextStep = model.weights[feature] - step * weightGradient[feature];
            model.weights[feature] = nextStep;
            if (useMomentum && epoch > 0) {
                model.weights[feature] += optimizer.momentum * (nextStep - previousStep.weights[feature]);
            }
            previousStep.weights[feature] = nextStep;
        }
        const double nextBiasStep = model.bias - step * biasGradient;
        model.bias = nextBiasStep;
        // Match upstream: the first epoch is an ordinary gradient step.
        if (useMomentum && epoch > 0) {
            model.bias += optimizer.momentum * (nextBiasStep - previousStep.bias);
        }
        previousStep.bias = nextBiasStep;
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
