// Row-major zero padding and label replication adapted from utils.cpp and
// enc_matrix.h in openfheorg/openfhe-logreg-training-examples.
// Copyright (c) 2023, Duality Technologies Inc. All rights reserved.
// The upstream BSD-2-Clause notice is retained in THIRD_PARTY_NOTICES.md.

#include "openfhe_lab/sample_packing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace labfhe {

std::size_t PackedRowWidth(std::size_t featureCount) {
    if (featureCount == 0) {
        throw std::invalid_argument("Cannot pack zero features");
    }
    std::size_t width = 1;
    while (width < featureCount) {
        if (width > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::invalid_argument("Feature count is too large");
        }
        width *= 2;
    }
    return width;
}

std::vector<PackedTrainingBlock> PackTrainingData(
    const labml::Dataset& data,
    std::size_t slots,
    std::size_t rowWidth) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot pack an empty training dataset");
    }
    if (slots == 0 || (slots & (slots - 1)) != 0 || rowWidth == 0 ||
        (rowWidth & (rowWidth - 1)) != 0 || rowWidth > slots) {
        throw std::invalid_argument("Slots and row width must be compatible powers of two");
    }
    const auto featureCount = data.front().features.size();
    if (featureCount == 0 || featureCount > rowWidth) {
        throw std::invalid_argument("Features do not fit in a packed row");
    }
    const std::size_t capacity = slots / rowWidth;
    std::vector<PackedTrainingBlock> blocks;
    for (std::size_t start = 0; start < data.size(); start += capacity) {
        PackedTrainingBlock block{
            std::vector<double>(slots, 0.0),
            std::vector<double>(slots, 0.0),
            std::vector<double>(slots, 0.0),
            std::min(capacity, data.size() - start)};
        for (std::size_t row = 0; row < block.sampleCount; ++row) {
            const auto& sample = data[start + row];
            if (sample.features.size() != featureCount ||
                (sample.label != 0.0 && sample.label != 1.0)) {
                throw std::invalid_argument("Training rows must have equal widths and binary labels");
            }
            for (std::size_t col = 0; col < rowWidth; ++col) {
                const auto index = row * rowWidth + col;
                if (col < featureCount) {
                    if (!std::isfinite(sample.features[col])) {
                        throw std::invalid_argument("Training features must be finite");
                    }
                    block.features[index] = sample.features[col];
                }
                block.labels[index] = sample.label;
                block.validRows[index] = 1.0;
            }
        }
        blocks.push_back(std::move(block));
    }
    return blocks;
}

}  // namespace labfhe
