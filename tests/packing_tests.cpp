#include "openfhe_lab/sample_packing.hpp"
#include "openfhe_lab/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
    Require(rejected, "Invalid packing input was accepted");
}

void CheckLayoutAndGradient(const labml::Dataset& data) {
    const auto features = labml::FeatureCount(data);
    const auto width = labfhe::PackedRowWidth(features);
    constexpr std::size_t slots = 2048;
    const auto capacity = slots / width;
    const auto blocks = labfhe::PackTrainingData(data, slots, width);
    Require(blocks.size() == (data.size() + capacity - 1) / capacity, "Wrong block count");
    std::size_t seen = 0;
    for (const auto& block : blocks) {
        Require(block.sampleCount == std::min(capacity, data.size() - seen), "Wrong row count");
        for (std::size_t row = 0; row < capacity; ++row) {
            const bool valid = row < block.sampleCount;
            for (std::size_t col = 0; col < width; ++col) {
                const auto index = row * width + col;
                Require(block.validRows[index] == (valid ? 1.0 : 0.0), "Bad occupancy mask");
                Require(block.labels[index] == (valid ? data[seen + row].label : 0.0),
                        "Labels must be repeated across each row");
                Require(block.features[index] ==
                            (valid && col < features ? data[seen + row].features[col] : 0.0),
                        "Features must be row-major with zero-padded columns/rows");
            }
        }
        seen += block.sampleCount;
    }
    Require(seen == data.size(), "Packing lost or duplicated samples");

    // Independent plaintext execution of the packed full-batch circuit. A
    // non-zero bias after epoch one exercises the final-block occupancy mask.
    labml::PlainModel model{std::vector<double>(features, 0.0), 0.0};
    const auto reference = labml::TrainPlaintext(data, data, 4, 0.01);
    for (std::size_t epoch = 0; epoch < 4; ++epoch) {
        std::vector<double> gradient(features, 0.0);
        double biasGradient = 0.0;
        for (const auto& block : blocks) {
            for (std::size_t row = 0; row < capacity; ++row) {
                const auto offset = row * width;
                double score = model.bias;
                for (std::size_t col = 0; col < features; ++col) {
                    score += block.features[offset + col] * model.weights[col];
                }
                const double error = labml::PolynomialSigmoid(score) - block.labels[offset];
                for (std::size_t col = 0; col < features; ++col) {
                    gradient[col] += block.features[offset + col] * error;
                }
                biasGradient += error * block.validRows[offset];
            }
        }
        for (std::size_t col = 0; col < features; ++col) {
            model.weights[col] -= (0.01 / data.size()) * gradient[col];
        }
        model.bias -= (0.01 / data.size()) * biasGradient;
        Require(labml::MaximumModelError(model, reference.epochs[epoch].model) < 1e-12,
                "Packing changed the full-batch update");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 3, "Expected the two lab dataset paths");
        const auto logreg = labml::LabTrainTestSplit(labml::LoadLogRegSample(argv[1])).train;
        const auto framingham = labml::LabTrainTestSplit(labml::LoadAndPrepareFramingham(argv[2])).train;
        Require(labfhe::PackedRowWidth(2) == 2 && labfhe::PackedRowWidth(9) == 16,
                "Wrong power-of-two feature padding");
        CheckLayoutAndGradient(logreg);
        CheckLayoutAndGradient(framingham);
        for (const std::size_t count : {1U, 127U, 128U, 129U}) {
            CheckLayoutAndGradient(labml::Dataset(framingham.begin(), framingham.begin() + count));
        }
        RequireInvalid([&] { labfhe::PackTrainingData({}, 2048, 16); });
        RequireInvalid([&] { labfhe::PackTrainingData(framingham, 2048, 8); });
        RequireInvalid([&] { labfhe::PackTrainingData(framingham, 2048, 9); });
        auto invalid = labml::Dataset(logreg.begin(), logreg.begin() + 2);
        invalid[1].features.pop_back();
        RequireInvalid([&] { labfhe::PackTrainingData(invalid, 2048, 2); });
        invalid = {logreg.front()};
        invalid[0].features[0] = std::numeric_limits<double>::quiet_NaN();
        RequireInvalid([&] { labfhe::PackTrainingData(invalid, 2048, 2); });
        std::cout << "Packing layout, padding, and full-batch gradient tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "packing test failure: " << error.what() << '\n';
        return 1;
    }
}
