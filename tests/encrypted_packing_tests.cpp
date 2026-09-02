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

void CheckEncrypted(
    const labml::Dataset& data,
    std::size_t count,
    bool checkNesterov,
    labml::SigmoidApproximation sigmoid) {
    // Only subsets of the user's two lab datasets, not replacement datasets.
    const auto split = labml::LabTrainTestSplit(data);
    const labml::Dataset train(split.train.begin(), split.train.begin() + count);
    labfhe::CkksConfiguration configuration;
    configuration.sigmoid = sigmoid;
    configuration.rowWidth = static_cast<std::uint32_t>(
        labfhe::PackedRowWidth(labml::FeatureCount(train)));
    const auto runtime = labfhe::CreateFheRuntime(configuration);
    Require(runtime.sigmoid == sigmoid && runtime.multiplicativeDepth ==
                (sigmoid == labml::SigmoidApproximation::Chebyshev ? 35U : 29U),
            "Runtime must select the sigmoid circuit and its default CKKS depth");
    const auto encrypted = labfhe::EncryptDataset(runtime, train);
    const auto capacity = runtime.slots / runtime.rowWidth;
    Require(encrypted.blocks.size() == (count + capacity - 1) / capacity,
            "Unexpected encrypted block count");
    // Cover both refresh modes and packing shapes for each approximation,
    // plus high-momentum NAG with real refresh. Plaintext tests cover the
    // full NAG matrix.
    std::vector<labml::OptimizerConfiguration> optimizers{
        labml::OptimizerConfiguration{}};
    if (checkNesterov) {
        optimizers.push_back(
            {labml::Optimizer::NesterovAcceleratedGradient, 0.8});
    }
    // Cubic GD first bootstraps in epoch 3; Chebyshev in epoch 2. In both
    // cases include a further epoch to exercise the refreshed model.
    const std::size_t epochs = sigmoid == labml::SigmoidApproximation::Chebyshev ? 3 : 4;
    for (const auto optimizer : optimizers) {
        const auto reference = labml::TrainPlaintext(train, split.test, epochs, 0.01, optimizer, sigmoid);
        auto mismatched = reference;
        mismatched.sigmoid = sigmoid == labml::SigmoidApproximation::Chebyshev
            ? labml::SigmoidApproximation::Cubic : labml::SigmoidApproximation::Chebyshev;
        bool rejected = false;
        try {
            labfhe::TrainEncrypted(runtime, encrypted, train, split.test, mismatched,
                                  epochs, 0.01, labfhe::RefreshMethod::RealBootstrapping, optimizer);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(rejected, "Encrypted trainer must reject a reference using a different sigmoid");
        std::cout << "  sigmoid " << labml::SigmoidApproximationName(sigmoid)
                  << ", optimizer " << labml::OptimizerName(optimizer.method)
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
                Require(result.epochs[epochs - 2].refreshed,
                        "Packed circuit must bootstrap before the last test epoch");
                Require(result.epochs.back().refreshed,
                        "Encrypted training must continue after the first real bootstrap");
            }
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 4, "Expected the two lab dataset paths and sigmoid approximation");
        const std::string selection = argv[3];
        Require(selection == "chebyshev" || selection == "cubic", "Unknown test sigmoid");
        const auto sigmoid = selection == "chebyshev"
            ? labml::SigmoidApproximation::Chebyshev : labml::SigmoidApproximation::Cubic;
        CheckEncrypted(labml::LoadLogRegSample(argv[1]), 13, true, sigmoid);
        // 128 rows fit in a 9-feature block; 129 exercises cross-block sums
        // and a heavily padded final block, especially its bias gradient.
        CheckEncrypted(labml::LoadAndPrepareFramingham(argv[2]), 129, false, sigmoid);
        std::cout << "Encrypted GD/NAG packing, refresh, and post-bootstrap continuation tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "encrypted packing test failure: " << error.what() << '\n';
        return 1;
    }
}
