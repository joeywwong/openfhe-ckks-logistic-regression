#include "openfhe_lab/ckks_logistic_regression.hpp"
#include "openfhe_lab/dataset.hpp"
#include "openfhe_lab/logistic_regression.hpp"
#include "openfhe_lab/sample_packing.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset{"all"};
    std::string refresh{"both"};
    std::size_t epochs{100};
    double learningRate{0.01};
    labml::OptimizerConfiguration optimizer;
    std::string outputPath;
    bool showHelp{false};
};

double ElapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void PrintUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --dataset logreg|framingham|all  Dataset selection (default: all)\n"
        << "  --refresh simulated|real|both    Model refresh method (default: both)\n"
        << "  --epochs N                       Lab default is 100\n"
        << "  --learning-rate X                Lab default is 0.01\n"
        << "  --optimizer gd|nag               Full-batch optimizer (default: gd)\n"
        << "  --momentum X                     NAG coefficient in [0, 1) (default: 0.1)\n"
        << "  --output PATH                    Per-epoch CSV output path\n"
        << "  --help                           Show this message\n";
}

Options ParseArguments(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--help") {
            options.showHelp = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value after " + flag);
        }
        const std::string value = argv[++index];
        if (flag == "--dataset") {
            options.dataset = value;
        }
        else if (flag == "--refresh") {
            options.refresh = value;
        }
        else if (flag == "--epochs") {
            options.epochs = static_cast<std::size_t>(std::stoull(value));
        }
        else if (flag == "--learning-rate") {
            options.learningRate = std::stod(value);
        }
        else if (flag == "--optimizer") {
            if (value == "gd") {
                options.optimizer.method = labml::Optimizer::GradientDescent;
            }
            else if (value == "nag") {
                options.optimizer.method = labml::Optimizer::NesterovAcceleratedGradient;
            }
            else {
                throw std::invalid_argument("--optimizer must be gd or nag");
            }
        }
        else if (flag == "--momentum") {
            std::size_t parsed = 0;
            options.optimizer.momentum = std::stod(value, &parsed);
            if (parsed != value.size()) {
                throw std::invalid_argument("--momentum must be a number in [0, 1)");
            }
        }
        else if (flag == "--output") {
            options.outputPath = value;
        }
        else {
            throw std::invalid_argument("Unknown option: " + flag);
        }
    }
    if (options.dataset != "logreg" && options.dataset != "framingham" && options.dataset != "all") {
        throw std::invalid_argument("--dataset must be logreg, framingham, or all");
    }
    if (options.refresh != "simulated" && options.refresh != "real" && options.refresh != "both") {
        throw std::invalid_argument("--refresh must be simulated, real, or both");
    }
    if (options.epochs == 0 || !std::isfinite(options.learningRate) || options.learningRate <= 0.0) {
        throw std::invalid_argument("Epochs and learning rate must be positive and finite");
    }
    labml::ValidateOptimizerConfiguration(options.optimizer);
    if (options.outputPath.empty()) {
        options.outputPath = std::string(OPENFHE_LAB_SOURCE_DIR) + "/results/" +
            (options.optimizer.method == labml::Optimizer::NesterovAcceleratedGradient
                ? "benchmark_nag.csv" : "benchmark_packed.csv");
    }
    return options;
}

std::vector<labml::DatasetKind> SelectedDatasets(const Options& options) {
    if (options.dataset == "logreg") {
        return {labml::DatasetKind::LogRegSample};
    }
    if (options.dataset == "framingham") {
        return {labml::DatasetKind::Framingham};
    }
    return {labml::DatasetKind::LogRegSample, labml::DatasetKind::Framingham};
}

std::vector<labfhe::RefreshMethod> SelectedRefreshMethods(const Options& options) {
    if (options.refresh == "simulated") {
        return {labfhe::RefreshMethod::SimulatedBootstrapping};
    }
    if (options.refresh == "real") {
        return {labfhe::RefreshMethod::RealBootstrapping};
    }
    return {
        labfhe::RefreshMethod::SimulatedBootstrapping,
        labfhe::RefreshMethod::RealBootstrapping,
    };
}

labml::Dataset LoadDataset(labml::DatasetKind kind) {
    const std::string sourceDirectory = OPENFHE_LAB_SOURCE_DIR;
    if (kind == labml::DatasetKind::LogRegSample) {
        return labml::LoadLogRegSample(sourceDirectory + "/data/LogReg_sample_dataset.csv");
    }
    return labml::LoadAndPrepareFramingham(sourceDirectory + "/data/framingham.csv");
}

void WriteCsvHeader(std::ofstream& output) {
    output << "dataset,method,epoch,homomorphic_seconds,refresh_seconds,metric_decryption_seconds,"
              "paired_simulated_refresh_seconds,seconds_per_epoch,accuracy,loss,"
              "max_plaintext_model_error,refreshed,level_before_refresh,level_after_refresh,optimizer,momentum\n";
}

void WriteOptimizerColumns(std::ofstream& output, const labml::OptimizerConfiguration& optimizer) {
    output << ',' << labml::OptimizerName(optimizer.method) << ','
           << (optimizer.method == labml::Optimizer::NesterovAcceleratedGradient ? optimizer.momentum : 0.0)
           << '\n';
}

void WritePlaintextRows(
    std::ofstream& output,
    const std::string& dataset,
    const labml::PlaintextTrainingResult& result) {
    for (const auto& epoch : result.epochs) {
        output << dataset << ",plaintext," << epoch.epoch << ',' << epoch.seconds
               << ",0,0,0," << epoch.seconds << ',' << epoch.accuracy << ',' << epoch.loss
               << ",0,0,0,0";
        WriteOptimizerColumns(output, result.optimizer);
    }
}

void WriteEncryptedRows(
    std::ofstream& output,
    const std::string& dataset,
    labfhe::RefreshMethod method,
    const labml::OptimizerConfiguration& optimizer,
    const labfhe::EncryptedTrainingResult& result) {
    for (const auto& epoch : result.epochs) {
        output << dataset << ',' << labfhe::RefreshMethodName(method) << ',' << epoch.epoch << ','
               << epoch.homomorphicSeconds << ',' << epoch.refreshSeconds << ','
               << epoch.metricDecryptionSeconds << ',' << epoch.pairedSimulatedRefreshSeconds << ','
               << epoch.secondsPerEpoch << ','
               << epoch.accuracy << ',' << epoch.loss << ',' << epoch.maximumPlaintextModelError
               << ',' << (epoch.refreshed ? 1 : 0) << ',' << epoch.levelBeforeRefresh << ','
               << epoch.levelAfterRefresh;
        WriteOptimizerColumns(output, optimizer);
    }
}

double MeanRefreshSeconds(const labfhe::EncryptedTrainingResult& result) {
    double total = 0.0;
    std::size_t count = 0;
    for (const auto& epoch : result.epochs) {
        if (epoch.refreshed) {
            total += epoch.refreshSeconds;
            ++count;
        }
    }
    return count == 0 ? 0.0 : total / static_cast<double>(count);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = ParseArguments(argc, argv);
        if (options.showHelp) {
            PrintUsage(argv[0]);
            return 0;
        }

        const std::filesystem::path outputPath(options.outputPath);
        if (outputPath.has_parent_path()) {
            std::filesystem::create_directories(outputPath.parent_path());
        }
        std::ofstream output(outputPath);
        if (!output) {
            throw std::runtime_error("Could not open result file: " + options.outputPath);
        }
        output << std::setprecision(10);
        WriteCsvHeader(output);

        std::cout << "OpenFHE port of the TenSEAL encrypted logistic-regression lab\n"
                  << "Fixed settings: shuffled 70/30 split, seed 4, degree-59 Chebyshev sigmoid, "
                     "full-batch training\n"
                  << "Optimizer: " << labml::OptimizerName(options.optimizer.method)
                  << ", momentum: " << (options.optimizer.method == labml::Optimizer::NesterovAcceleratedGradient
                      ? options.optimizer.momentum : 0.0) << "\n"
                  << "Epochs: " << options.epochs << ", learning rate: " << options.learningRate << "\n"
                  << "Result CSV: " << options.outputPath << "\n\n";

        for (const auto datasetKind : SelectedDatasets(options)) {
            const std::string datasetName = labml::DatasetName(datasetKind);
            std::cout << "Dataset: " << datasetName << '\n';
            const auto data = LoadDataset(datasetKind);
            const auto split = labml::LabTrainTestSplit(data, 0.30, 4);
            std::cout << "  " << data.size() << " samples, " << labml::FeatureCount(data)
                      << " features, " << split.train.size() << " train / " << split.test.size()
                      << " test\n";

            const auto plaintext = labml::TrainPlaintext(
                split.train, split.test, options.epochs, options.learningRate, options.optimizer);
            WritePlaintextRows(output, datasetName, plaintext);
            const auto& plainFinal = plaintext.epochs.back();
            std::cout << "  plaintext final: accuracy " << plainFinal.accuracy << ", loss "
                      << plainFinal.loss << "\n";

            const auto setupStart = Clock::now();
            labfhe::CkksConfiguration configuration;
            configuration.rowWidth = static_cast<std::uint32_t>(
                labfhe::PackedRowWidth(labml::FeatureCount(split.train)));
            const auto runtime = labfhe::CreateFheRuntime(configuration);
            const double setupSeconds = ElapsedSeconds(setupStart);
            std::cout << "  OpenFHE setup: " << setupSeconds << " s; ring dimension "
                      << runtime.context->GetRingDimension() << ", slots " << runtime.slots
                      << ", row width " << runtime.rowWidth
                      << ", bootstrap slots " << runtime.bootstrapSlots
                      << ", multiplicative depth " << runtime.multiplicativeDepth << '\n';

            const auto encryptionStart = Clock::now();
            const auto encryptedTrain = labfhe::EncryptDataset(runtime, split.train);
            const double encryptionSeconds = ElapsedSeconds(encryptionStart);
            std::cout << "  packed " << encryptedTrain.sampleCount << " training samples into "
                      << encryptedTrain.blocks.size() << " blocks ("
                      << 2 * encryptedTrain.blocks.size() << " ciphertexts; "
                      << runtime.slots / runtime.rowWidth << " rows/block) in "
                      << encryptionSeconds << " s\n";

            for (const auto method : SelectedRefreshMethods(options)) {
                std::cout << "  Method: " << labfhe::RefreshMethodName(method) << '\n';
                const auto encrypted = labfhe::TrainEncrypted(
                    runtime,
                    encryptedTrain,
                    split.train,
                    split.test,
                    plaintext,
                    options.epochs,
                    options.learningRate,
                    method,
                    options.optimizer);
                WriteEncryptedRows(output, datasetName, method, options.optimizer, encrypted);
                output.flush();
                const auto& finalEpoch = encrypted.epochs.back();
                std::cout << std::setprecision(6) << "    final accuracy " << finalEpoch.accuracy << ", loss "
                          << finalEpoch.loss << ", mean refresh " << MeanRefreshSeconds(encrypted)
                          << " s, max model error " << std::scientific << finalEpoch.maximumPlaintextModelError
                          << std::defaultfloat << "\n";
            }
            std::cout << '\n';
        }

        std::cout << "Benchmark complete. Per-epoch measurements written to " << options.outputPath << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
