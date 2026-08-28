#include "openfhe_lab/ckks_logistic_regression.hpp"
#include "openfhe_lab/sample_packing.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void CheckEncrypted(const labml::Dataset& data, std::size_t count) {
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
    // Reuse the same packed inputs/context for both optimizers. Non-default
    // momentum stresses the retained state across multiple real bootstraps.
    for (const auto optimizer : {
             labml::OptimizerConfiguration{},
             labml::OptimizerConfiguration{labml::Optimizer::NesterovAcceleratedGradient, 0.1},
             labml::OptimizerConfiguration{labml::Optimizer::NesterovAcceleratedGradient, 0.8},
             labml::OptimizerConfiguration{labml::Optimizer::NesterovAcceleratedGradient, 0.0}}) {
        const auto reference = labml::TrainPlaintext(train, split.test, 4, 0.01, optimizer);
        std::cout << "  optimizer " << labml::OptimizerName(optimizer.method)
                  << ", momentum " << optimizer.momentum << '\n';
        for (const auto method : {labfhe::RefreshMethod::SimulatedBootstrapping,
                                  labfhe::RefreshMethod::RealBootstrapping}) {
            const auto result = labfhe::TrainEncrypted(
                runtime, encrypted, train, split.test, reference, 4, 0.01, method, optimizer);
            Require(result.epochs.size() == 4, "Encrypted trainer must report every epoch");
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
            Require(refreshes > 0, "Four-epoch integration test never exercised refresh");
            if (method == labfhe::RefreshMethod::RealBootstrapping) {
                Require(result.epochs[2].refreshed,
                        "Packed circuit should bootstrap by epoch 3, then test continuation at epoch 4");
            }
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 3, "Expected the two lab dataset paths");
        CheckEncrypted(labml::LoadLogRegSample(argv[1]), 13);
        // 128 rows fit in a 9-feature block; 129 exercises cross-block sums
        // and a heavily padded final block, especially its bias gradient.
        CheckEncrypted(labml::LoadAndPrepareFramingham(argv[2]), 129);
        std::cout << "Encrypted GD/NAG packing, refresh, and post-bootstrap continuation tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "encrypted packing test failure: " << error.what() << '\n';
        return 1;
    }
}
