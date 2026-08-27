#pragma once

#include "openfhe_lab/dataset.hpp"

#include <cstddef>
#include <vector>

namespace labfhe {

// One power-of-two-width row per sample, as in OpenFHE's Mat2CtMRM.
// The last block is zero-padded; validRows excludes padding from bias gradients.
struct PackedTrainingBlock {
    std::vector<double> features;
    std::vector<double> labels;
    std::vector<double> validRows;
    std::size_t sampleCount{};
};

std::size_t PackedRowWidth(std::size_t featureCount);

std::vector<PackedTrainingBlock> PackTrainingData(
    const labml::Dataset& data,
    std::size_t slots,
    std::size_t rowWidth);

}  // namespace labfhe
