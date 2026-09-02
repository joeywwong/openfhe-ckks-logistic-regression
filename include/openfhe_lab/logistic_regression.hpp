#pragma once

#include "openfhe_lab/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace labml {

enum class Optimizer {
    GradientDescent,
    NesterovAcceleratedGradient,
};

struct OptimizerConfiguration {
    Optimizer method{Optimizer::GradientDescent};
    double momentum{0.1};  // Fixed coefficient used by the upstream NAG example.
};

std::string OptimizerName(Optimizer optimizer);
void ValidateOptimizerConfiguration(const OptimizerConfiguration& configuration);

enum class SigmoidApproximation {
    Cubic,
    Chebyshev,
};

std::string SigmoidApproximationName(SigmoidApproximation approximation);

struct PlainModel {
    std::vector<double> weights;
    double bias{};
};

struct PlaintextEpochMetrics {
    std::size_t epoch{};
    double seconds{};
    double accuracy{};
    double loss{};
    PlainModel model;
};

struct PlaintextTrainingResult {
    PlainModel finalModel;
    std::vector<PlaintextEpochMetrics> epochs;
    OptimizerConfiguration optimizer;
    SigmoidApproximation sigmoid{SigmoidApproximation::Cubic};
};

// Match the official OpenFHE logistic-regression example: approximate the
// logistic function with a degree-59 Chebyshev interpolant over [-16, 16].
inline constexpr double kSigmoidApproximationLowerBound{-16.0};
inline constexpr double kSigmoidApproximationUpperBound{16.0};
inline constexpr std::uint32_t kSigmoidApproximationDegree{59};

// Evaluate either the Chebyshev series used by OpenFHE's EvalLogistic or the
// lab/main-branch cubic 0.5 + 0.197*x - 0.004*x^3, matching the encrypted circuit.
double PolynomialSigmoid(
    double score,
    SigmoidApproximation approximation = SigmoidApproximation::Cubic);

// The lab reported cross-entropy with the exact sigmoid after decrypting the
// model; this function is therefore deliberately separate from the circuit.
double ExactSigmoid(double score);

double LinearScore(const PlainModel& model, const Sample& sample);
double Accuracy(const PlainModel& model, const Dataset& data);
double ExactLogLoss(const PlainModel& model, const Dataset& data);

PlaintextTrainingResult TrainPlaintext(
    const Dataset& train,
    const Dataset& test,
    std::size_t epochs,
    double learningRate,
    const OptimizerConfiguration& optimizer = {},
    SigmoidApproximation sigmoid = SigmoidApproximation::Cubic);

double MaximumModelError(const PlainModel& first, const PlainModel& second);

}  // namespace labml
