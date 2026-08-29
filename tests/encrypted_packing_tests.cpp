#include "openfhe_lab/ckks_logistic_regression.hpp"
#include "openfhe_lab/sample_packing.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void CheckEncrypted(const labml::Dataset& data, std::size_t count, bool checkNesterov) {
    // Only subsets of the user's two lab datasets, not replacement datasets.
    const auto split = labml::LabTrainTestSplit(data);
    const labml::Dataset train(split.train.begin(), split.train.begin() + count);
    labfhe::CkksConfiguration configuration;
    configuration.rowWidth = static_cast<std::uint32_t>(
        labfhe::PackedRowWidth(labml::FeatureCount(train)));
    const auto runtime = labfhe::CreateFheRuntime(configuration);
    const auto encrypted = labfhe::EncryptDataset(runtime, train);
    const auto capacity = runtime.slots / runtime.rowWidth;
    Require(encrypted.blocks.size() == (count + capacity - 1) / capacity,
            "Unexpected encrypted block count");
    // Degree-59 EvalLogistic makes each encrypted epoch much more expensive
    // than the former cubic. Keep both refresh modes for GD on both packing
    // shapes, plus one high-momentum real-bootstrap case to preserve encrypted
    // NAG state/continuation coverage. Plaintext tests cover the full NAG matrix.
    std::vector<labml::OptimizerConfiguration> optimizers{
        labml::OptimizerConfiguration{}};
    if (checkNesterov) {
        optimizers.push_back(
            {labml::Optimizer::NesterovAcceleratedGradient, 0.8});
    }
    constexpr std::size_t epochs = 3;
    for (const auto optimizer : optimizers) {
        const auto reference = labml::TrainPlaintext(train, split.test, epochs, 0.01, optimizer);
        std::cout << "  optimizer " << labml::OptimizerName(optimizer.method)
                  << ", momentum " << optimizer.momentum << '\n';
        for (const auto method : {labfhe::RefreshMethod::SimulatedBootstrapping,
                                  labfhe::RefreshMethod::RealBootstrapping}) {
            if (optimizer.method == labml::Optimizer::NesterovAcceleratedGradient &&
                method == labfhe::RefreshMethod::SimulatedBootstrapping) {
                continue;
            }
            const auto result = labfhe::TrainEncrypted(
                runtime, encrypted, train, split.test, reference, epochs, 0.01, method, optimizer);
            Require(result.epochs.size() == epochs, "Encrypted trainer must report every epoch");
            std::size_t refreshes = 0;
            for (std::size_t index = 0; index < result.epochs.size(); ++index) {
                const auto& epoch = result.epochs[index];
                Require(std::isfinite(epoch.maximumPlaintextModelError) &&
                            epoch.maximumPlaintextModelError < 1e-5,
                        "Packed encrypted model differs from the matching plaintext optimizer");
                Require(std::abs(epoch.loss - reference.epochs[index].loss) < 1e-6,
                        "Packed encrypted loss differs from plaintext");
                Require(std::abs(epoch.accuracy - reference.epochs[index].accuracy) < 1e-12,
                        "Packed encrypted accuracy differs from plaintext");
                if (epoch.refreshed) {
                    ++refreshes;
                    Require(epoch.levelAfterRefresh < epoch.levelBeforeRefresh,
                            "Refresh did not restore levels");
                    Require(epoch.refreshSeconds > 0.0, "Refresh was not timed");
                }
                if (method == labfhe::RefreshMethod::SimulatedBootstrapping) {
                    Require(epoch.refreshed && epoch.levelAfterRefresh == 0,
                            "Simulated refresh must produce fresh ciphertexts each epoch");
                }
            }
            Require(refreshes > 0, "Encrypted integration test never exercised refresh");
            if (method == labfhe::RefreshMethod::RealBootstrapping) {
                Require(result.epochs[1].refreshed,
                        "Degree-59 packed circuit should bootstrap by epoch 2");
                Require(result.epochs[2].refreshed,
                        "Encrypted training must continue after the first real bootstrap");
            }
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 3, "Expected the two lab dataset paths");
        CheckEncrypted(labml::LoadLogRegSample(argv[1]), 13, true);
        // 128 rows fit in a 9-feature block; 129 exercises cross-block sums
        // and a heavily padded final block, especially its bias gradient.
        CheckEncrypted(labml::LoadAndPrepareFramingham(argv[2]), 129, false);
        std::cout << "Encrypted GD/NAG packing, refresh, and post-bootstrap continuation tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "encrypted packing test failure: " << error.what() << '\n';
        return 1;
    }
}
