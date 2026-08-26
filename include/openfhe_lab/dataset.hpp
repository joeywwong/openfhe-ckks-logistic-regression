#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace labml {

struct Sample {
    std::vector<double> features;
    double label{};
};

using Dataset = std::vector<Sample>;

struct DatasetSplit {
    Dataset train;
    Dataset test;
};

enum class DatasetKind {
    LogRegSample,
    Framingham,
};

std::string DatasetName(DatasetKind kind);

// Loads the two-feature, linearly separable CSV exactly as a binary dataset.
Dataset LoadLogRegSample(const std::string& path);

// Reproduces the lab's heart_disease_data() transformation: drop every row
// containing a missing value, remove the same six columns, balance both labels
// with seed 73, and standardize the resulting nine features over the complete
// balanced dataset before the train/test split.
Dataset LoadAndPrepareFramingham(const std::string& path);

// Reproduces split_train_test(): shuffle once and use the first floor(30%) as
// test data. It is intentionally not stratified because the lab was not.
DatasetSplit LabTrainTestSplit(
    const Dataset& data,
    double testRatio = 0.30,
    std::uint32_t seed = 4);

std::size_t FeatureCount(const Dataset& data);

}  // namespace labml
