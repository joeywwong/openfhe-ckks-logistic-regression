#include "openfhe_lab/dataset.hpp"
#include "openfhe_lab/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void RequireInvalid(Function function) {
    bool rejected = false;
    try {
        function();
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }
    Require(rejected, "Invalid training configuration was accepted");
}

void CheckChebyshevSigmoid() {
    Require(labml::kSigmoidApproximationLowerBound == -16.0 &&
                labml::kSigmoidApproximationUpperBound == 16.0 &&
                labml::kSigmoidApproximationDegree == 59,
            "Sigmoid approximation must match the official OpenFHE example");

    double maximumError = 0.0;
    for (int index = 0; index <= 3200; ++index) {
        const double score = labml::kSigmoidApproximationLowerBound +
            (labml::kSigmoidApproximationUpperBound -
             labml::kSigmoidApproximationLowerBound) *
                static_cast<double>(index) / 3200.0;
        const double approximate = labml::PolynomialSigmoid(score, labml::SigmoidApproximation::Chebyshev);
        maximumError = std::max(
            maximumError, std::abs(approximate - labml::ExactSigmoid(score)));
        Require(std::abs(
                    approximate + labml::PolynomialSigmoid(-score, labml::SigmoidApproximation::Chebyshev) - 1.0) < 1e-12,
                "Chebyshev sigmoid must preserve logistic symmetry");
    }
    Require(maximumError < 6e-6,
            "Degree-59 Chebyshev sigmoid exceeded its expected error on [-16, 16]");
}

void CheckCubicSigmoid() {
    // Fixed values from the original lab polynomial, including values outside
    // [0, 1]: the training approximation must not be clamped.
    for (const auto& point : std::vector<std::pair<double, double>>{
             {0.0, 0.5}, {1.0, 0.693}, {-1.0, 0.307}, {2.0, 0.862},
             {-2.0, 0.138}, {4.0, 1.032}, {-4.0, -0.032}, {10.0, -1.53}}) {
        Require(std::abs(labml::PolynomialSigmoid(point.first, labml::SigmoidApproximation::Cubic) -
                         point.second) < 1e-12,
                "Cubic sigmoid must reproduce the original lab polynomial");
    }
    RequireInvalid([] {
        labml::PolynomialSigmoid(0.0, static_cast<labml::SigmoidApproximation>(-1));
    });
}

void CheckNesterovUpdates(const labml::Dataset& data, labml::SigmoidApproximation sigmoid) {
    constexpr std::size_t epochs = 5;
    constexpr double learningRate = 0.01;
    const auto gd = labml::TrainPlaintext(data, data, epochs, learningRate, {}, sigmoid);
    const labml::OptimizerConfiguration zeroMomentum{labml::Optimizer::NesterovAcceleratedGradient, 0.0};
    const auto zero = labml::TrainPlaintext(data, data, epochs, learningRate, zeroMomentum, sigmoid);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        Require(labml::MaximumModelError(gd.epochs[epoch].model, zero.epochs[epoch].model) == 0.0,
                "Zero-momentum NAG must reduce exactly to GD");
    }

    for (const double momentum : {0.1, 0.8}) {
        const labml::OptimizerConfiguration optimizer{labml::Optimizer::NesterovAcceleratedGradient, momentum};
        const auto nag = labml::TrainPlaintext(data, data, epochs, learningRate, optimizer, sigmoid);
        Require(nag.epochs.size() == epochs, "NAG must report every epoch");
        Require(labml::MaximumModelError(nag.epochs.front().model, gd.epochs.front().model) == 0.0,
                "Upstream NAG starts with one ordinary gradient step");

        // Independent velocity formulation after the common first step:
        // lookAhead = base + mu*v; v = mu*v - lr*g(lookAhead); base += v.
        // The reported theta is the next lookAhead, not the unaccelerated base.
        auto base = gd.epochs.front().model;
        labml::PlainModel velocity{std::vector<double>(base.weights.size(), 0.0), 0.0};
        for (std::size_t epoch = 1; epoch < epochs; ++epoch) {
            auto lookAhead = base;
            for (std::size_t feature = 0; feature < base.weights.size(); ++feature) {
                lookAhead.weights[feature] += momentum * velocity.weights[feature];
            }
            lookAhead.bias += momentum * velocity.bias;
            labml::PlainModel gradient{std::vector<double>(base.weights.size(), 0.0), 0.0};
            for (const auto& sample : data) {
                const double error = labml::PolynomialSigmoid(labml::LinearScore(lookAhead, sample), sigmoid) - sample.label;
                for (std::size_t feature = 0; feature < base.weights.size(); ++feature) {
                    gradient.weights[feature] += sample.features[feature] * error / data.size();
                }
                gradient.bias += error / data.size();
            }
            for (std::size_t feature = 0; feature < base.weights.size(); ++feature) {
                velocity.weights[feature] = momentum * velocity.weights[feature] -
                    learningRate * gradient.weights[feature];
                base.weights[feature] += velocity.weights[feature];
                lookAhead.weights[feature] = base.weights[feature] + momentum * velocity.weights[feature];
            }
            velocity.bias = momentum * velocity.bias - learningRate * gradient.bias;
            base.bias += velocity.bias;
            lookAhead.bias = base.bias + momentum * velocity.bias;
            Require(labml::MaximumModelError(lookAhead, nag.epochs[epoch].model) < 1e-12,
                    "NAG differs from the independent look-ahead velocity update");
            Require(std::abs(nag.epochs[epoch].loss - labml::ExactLogLoss(lookAhead, data)) < 1e-12,
                    "NAG metrics must use the reported look-ahead model");
        }
        Require(labml::MaximumModelError(nag.finalModel, gd.finalModel) > 1e-8,
                "Nonzero momentum must change the training trajectory");
        Require(std::abs(nag.finalModel.bias - gd.finalModel.bias) > 1e-8,
                "Nesterov acceleration must also update the bias");
        Require(labml::MaximumModelError(nag.finalModel, nag.epochs.back().model) == 0.0,
                "Final NAG model must match its last epoch");
    }

    for (const double momentum : {-0.1, 1.0, std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
        RequireInvalid([&] {
            labml::TrainPlaintext(data, data, epochs, learningRate,
                                 {labml::Optimizer::NesterovAcceleratedGradient, momentum});
        });
    }
    for (const double rate : {0.0, -0.01, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
        RequireInvalid([&] { labml::TrainPlaintext(data, data, epochs, rate); });
    }
    RequireInvalid([&] {
        labml::TrainPlaintext(data, data, epochs, learningRate, {static_cast<labml::Optimizer>(-1), 0.1});
    });
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 3, "Expected LogReg and Framingham CSV paths");
        CheckChebyshevSigmoid();
        CheckCubicSigmoid();
        Require(std::abs(labml::PolynomialSigmoid(0.0) - 0.5) < 1e-12,
                "Default cubic sigmoid must map zero to 0.5");
        Require(std::abs(labml::ExactSigmoid(0.0) - 0.5) < 1e-12,
                "Exact sigmoid must map zero to 0.5");

        const auto logreg = labml::LoadLogRegSample(argv[1]);
        Require(logreg.size() == 1000, "LogReg sample must contain 1000 records");
        Require(labml::FeatureCount(logreg) == 2, "LogReg sample must contain two features");
        const auto logregSplit = labml::LabTrainTestSplit(logreg, 0.30, 4);
        Require(logregSplit.train.size() == 700 && logregSplit.test.size() == 300,
                "LogReg split must reproduce the lab's 70/30 sizes");

        const auto framingham = labml::LoadAndPrepareFramingham(argv[2]);
        Require(framingham.size() == 1114, "Prepared Framingham data must contain 1114 records");
        Require(labml::FeatureCount(framingham) == 9, "Prepared Framingham data must contain nine features");
        const auto framinghamSplit = labml::LabTrainTestSplit(framingham, 0.30, 4);
        Require(framinghamSplit.train.size() == 780 && framinghamSplit.test.size() == 334,
                "Framingham split must reproduce the lab's 70/30 sizes");

        const auto trained = labml::TrainPlaintext(
            logregSplit.train, logregSplit.test, 2, 0.01);
        Require(trained.epochs.size() == 2, "Plaintext trainer must report every epoch");
        Require(std::isfinite(trained.epochs.back().loss), "Plaintext loss must be finite");
        Require(trained.epochs.back().accuracy >= 0.95,
                "Linearly separable LogReg sample should classify accurately");

        const auto cubic = labml::TrainPlaintext(
            logregSplit.train, logregSplit.test, 2, 0.01, {}, labml::SigmoidApproximation::Cubic);
        const auto chebyshev = labml::TrainPlaintext(
            logregSplit.train, logregSplit.test, 2, 0.01, {}, labml::SigmoidApproximation::Chebyshev);
        Require(trained.sigmoid == labml::SigmoidApproximation::Cubic &&
                    cubic.sigmoid == labml::SigmoidApproximation::Cubic &&
                    chebyshev.sigmoid == labml::SigmoidApproximation::Chebyshev,
                "Plaintext reference must record its sigmoid approximation, defaulting to cubic");
        Require(labml::MaximumModelError(trained.finalModel, cubic.finalModel) == 0.0,
                "Default training must match explicit cubic training");
        Require(labml::MaximumModelError(trained.finalModel, chebyshev.finalModel) > 1e-8,
                "Sigmoid selection must change the training trajectory");
        RequireInvalid([&] {
            labml::TrainPlaintext(logregSplit.train, logregSplit.test, 2, 0.01, {},
                                 static_cast<labml::SigmoidApproximation>(-1));
        });
        for (const auto sigmoid : {labml::SigmoidApproximation::Chebyshev,
                                   labml::SigmoidApproximation::Cubic}) {
            CheckNesterovUpdates(
                labml::Dataset(logregSplit.train.begin(), logregSplit.train.begin() + 13), sigmoid);
            CheckNesterovUpdates(
                labml::Dataset(framinghamSplit.train.begin(), framinghamSplit.train.begin() + 129), sigmoid);
        }

        std::cout << "All plaintext and Nesterov tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
