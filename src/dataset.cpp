#include "openfhe_lab/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace labml {
namespace {

std::vector<std::string> SplitCsvRow(const std::string& line) {
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) {
        if (!cell.empty() && cell.back() == '\r') {
            cell.pop_back();
        }
        cells.push_back(cell);
    }
    // std::getline omits a final empty cell.
    if (!line.empty() && line.back() == ',') {
        cells.emplace_back();
    }
    return cells;
}

double ParseFinite(const std::string& cell, std::size_t lineNumber) {
    try {
        std::size_t parsed = 0;
        const double value = std::stod(cell, &parsed);
        if (parsed != cell.size() || !std::isfinite(value)) {
            throw std::runtime_error("non-finite value");
        }
        return value;
    }
    catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric CSV value at line " + std::to_string(lineNumber));
    }
}

void ValidateBinaryLabel(double label, std::size_t lineNumber) {
    if (std::abs(label) > 1e-12 && std::abs(label - 1.0) > 1e-12) {
        throw std::runtime_error("Expected binary label 0 or 1 at line " + std::to_string(lineNumber));
    }
}

std::unordered_map<std::string, std::size_t> HeaderIndex(const std::vector<std::string>& header) {
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t index = 0; index < header.size(); ++index) {
        result.emplace(header[index], index);
    }
    return result;
}

bool IsMissingValue(const std::string& cell) {
    return cell.empty() || cell == "NA" || cell == "NaN" || cell == "nan";
}

}  // namespace

std::string DatasetName(DatasetKind kind) {
    switch (kind) {
        case DatasetKind::LogRegSample:
            return "LogReg_sample_dataset";
        case DatasetKind::Framingham:
            return "framingham";
    }
    throw std::invalid_argument("Unknown dataset kind");
}

Dataset LoadLogRegSample(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open dataset: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Dataset is empty: " + path);
    }
    const auto header = SplitCsvRow(line);
    if (header.size() != 3 || header[0] != "feature1" || header[1] != "feature2" ||
        header[2] != "label") {
        throw std::runtime_error("Unexpected LogReg_sample_dataset.csv header");
    }

    Dataset result;
    std::size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }
        const auto cells = SplitCsvRow(line);
        if (cells.size() != 3) {
            throw std::runtime_error("Unexpected CSV width at line " + std::to_string(lineNumber));
        }
        Sample sample;
        sample.features = {ParseFinite(cells[0], lineNumber), ParseFinite(cells[1], lineNumber)};
        sample.label    = ParseFinite(cells[2], lineNumber);
        ValidateBinaryLabel(sample.label, lineNumber);
        result.push_back(std::move(sample));
    }
    if (result.empty()) {
        throw std::runtime_error("Dataset contains no samples: " + path);
    }
    return result;
}

Dataset LoadAndPrepareFramingham(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open dataset: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Dataset is empty: " + path);
    }
    const auto header = SplitCsvRow(line);
    const auto columns = HeaderIndex(header);
    const std::vector<std::string> keptFeatures{
        "male",
        "age",
        "cigsPerDay",
        "prevalentStroke",
        "prevalentHyp",
        "totChol",
        "sysBP",
        "heartRate",
        "glucose",
    };
    for (const auto& required : keptFeatures) {
        if (columns.find(required) == columns.end()) {
            throw std::runtime_error("Missing Framingham column: " + required);
        }
    }
    const auto labelColumn = columns.find("TenYearCHD");
    if (labelColumn == columns.end()) {
        throw std::runtime_error("Missing Framingham column: TenYearCHD");
    }

    std::vector<std::vector<Sample>> byLabel(2);
    std::size_t lineNumber = 1;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty()) {
            continue;
        }
        const auto cells = SplitCsvRow(line);
        if (cells.size() != header.size()) {
            throw std::runtime_error("Unexpected CSV width at line " + std::to_string(lineNumber));
        }

        // pandas.DataFrame.dropna() in the lab ran before any columns were
        // removed, so a missing value in even a subsequently dropped column
        // removes the row here as well.
        bool hasMissingValue = false;
        for (const auto& cell : cells) {
            if (IsMissingValue(cell)) {
                hasMissingValue = true;
                break;
            }
        }
        if (hasMissingValue) {
            continue;
        }

        Sample sample;
        sample.features.reserve(keptFeatures.size());
        for (const auto& feature : keptFeatures) {
            sample.features.push_back(ParseFinite(cells[columns.at(feature)], lineNumber));
        }
        sample.label = ParseFinite(cells[labelColumn->second], lineNumber);
        ValidateBinaryLabel(sample.label, lineNumber);
        byLabel[static_cast<std::size_t>(sample.label)].push_back(std::move(sample));
    }

    if (byLabel[0].empty() || byLabel[1].empty()) {
        throw std::runtime_error("Framingham data must contain both labels");
    }
    const std::size_t balancedCount = std::min(byLabel[0].size(), byLabel[1].size());
    Dataset balanced;
    balanced.reserve(balancedCount * 2);
    for (auto& classSamples : byLabel) {
        // The lab invoked pandas.sample(random_state=73) separately for each
        // group, which is modeled by restarting the generator for each class.
        std::mt19937 generator(73);
        std::shuffle(classSamples.begin(), classSamples.end(), generator);
        balanced.insert(balanced.end(), classSamples.begin(), classSamples.begin() + balancedCount);
    }

    const std::size_t featureCount = keptFeatures.size();
    std::vector<double> means(featureCount, 0.0);
    for (const auto& sample : balanced) {
        for (std::size_t feature = 0; feature < featureCount; ++feature) {
            means[feature] += sample.features[feature];
        }
    }
    for (double& mean : means) {
        mean /= static_cast<double>(balanced.size());
    }

    // pandas.DataFrame.std() uses sample standard deviation (ddof=1).
    std::vector<double> standardDeviations(featureCount, 0.0);
    for (const auto& sample : balanced) {
        for (std::size_t feature = 0; feature < featureCount; ++feature) {
            const double difference = sample.features[feature] - means[feature];
            standardDeviations[feature] += difference * difference;
        }
    }
    for (double& deviation : standardDeviations) {
        deviation = std::sqrt(deviation / static_cast<double>(balanced.size() - 1));
        if (deviation == 0.0) {
            throw std::runtime_error("Cannot standardize a constant Framingham feature");
        }
    }
    for (auto& sample : balanced) {
        for (std::size_t feature = 0; feature < featureCount; ++feature) {
            sample.features[feature] =
                (sample.features[feature] - means[feature]) / standardDeviations[feature];
        }
    }
    return balanced;
}

DatasetSplit LabTrainTestSplit(const Dataset& data, double testRatio, std::uint32_t seed) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot split an empty dataset");
    }
    if (testRatio <= 0.0 || testRatio >= 1.0) {
        throw std::invalid_argument("testRatio must be strictly between zero and one");
    }

    std::vector<std::size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);

    const std::size_t testCount = static_cast<std::size_t>(
        std::floor(static_cast<double>(data.size()) * testRatio));
    DatasetSplit result;
    result.test.reserve(testCount);
    result.train.reserve(data.size() - testCount);
    for (std::size_t index = 0; index < indices.size(); ++index) {
        if (index < testCount) {
            result.test.push_back(data[indices[index]]);
        }
        else {
            result.train.push_back(data[indices[index]]);
        }
    }
    return result;
}

std::size_t FeatureCount(const Dataset& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot inspect an empty dataset");
    }
    const std::size_t featureCount = data.front().features.size();
    for (const auto& sample : data) {
        if (sample.features.size() != featureCount) {
            throw std::runtime_error("Dataset contains inconsistent feature counts");
        }
    }
    return featureCount;
}

}  // namespace labml
