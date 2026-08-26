#pragma once

#include "openfhe_lab/dataset.hpp"

#include <cstddef>
#include <vector>

namespace labml {

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
};

// The training circuit uses the same degree-three approximation as the lab.
double PolynomialSigmoid(double score);

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
    double learningRate);

double MaximumModelError(const PlainModel& first, const PlainModel& second);

}  // namespace labml
