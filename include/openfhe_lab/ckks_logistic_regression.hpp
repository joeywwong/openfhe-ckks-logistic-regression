#pragma once

#include "openfhe_lab/dataset.hpp"
#include "openfhe_lab/logistic_regression.hpp"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace labfhe {

enum class RefreshMethod {
    SimulatedBootstrapping,
    RealBootstrapping,
};

std::string RefreshMethodName(RefreshMethod method);

struct CkksConfiguration {
    // Full data packing in the existing 4096-degree demo ring. The periodic
    // weight/bias vectors still use the original 16-slot sparse bootstrap.
    std::uint32_t slots{2048};
    std::uint32_t rowWidth{16};
    std::uint32_t bootstrapSlots{16};
    std::vector<std::uint32_t> levelBudget{3, 3};
    // Zero selects the approximation's default: 16 for Chebyshev, 10 for cubic.
    // A nonzero value explicitly overrides the post-bootstrap level reserve.
    std::uint32_t levelsAvailableAfterBootstrap{0};
    labml::SigmoidApproximation sigmoid{labml::SigmoidApproximation::Cubic};
};

struct FheRuntime {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    std::uint32_t multiplicativeDepth{};
    std::uint32_t slots{};
    std::uint32_t rowWidth{};
    std::uint32_t bootstrapSlots{};
    std::shared_ptr<std::map<std::uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> sumRowsKeys;
    std::shared_ptr<std::map<std::uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> sumColsKeys;
    // EvalBootstrap in OpenFHE 1.1.2 returns the original ciphertext when it
    // still has at least as many towers as bootstrapping would return. A model
    // must be beyond this consumed level for refresh to be genuine.
    std::uint32_t bootstrapTriggerLevel{};
    labml::SigmoidApproximation sigmoid{labml::SigmoidApproximation::Cubic};
};

struct EncryptedBlock {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> features;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> labels;
    // Public row occupancy only; no feature/label values are exposed.
    lbcrypto::Plaintext validRows;
    std::size_t sampleCount{};
};

struct EncryptedDataset {
    std::vector<EncryptedBlock> blocks;
    std::size_t sampleCount{};
    std::size_t featureCount{};
};

struct EncryptedModel {
    // The lab encrypted weights and bias separately. This project deliberately
    // keeps that layout. NAG retains two such models (theta and previous phi).
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> weights;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> bias;
};

struct EncryptedEpochMetrics {
    std::size_t epoch{};
    double homomorphicSeconds{};
    double refreshSeconds{};
    double pairedSimulatedRefreshSeconds{};
    double metricDecryptionSeconds{};
    double secondsPerEpoch{};
    double accuracy{};
    double loss{};
    double maximumPlaintextModelError{};
    bool refreshed{};
    std::uint32_t levelBeforeRefresh{};
    std::uint32_t levelAfterRefresh{};
};

struct EncryptedTrainingResult {
    labml::PlainModel finalModel;
    std::vector<EncryptedEpochMetrics> epochs;
};

FheRuntime CreateFheRuntime(const CkksConfiguration& configuration = {});

EncryptedDataset EncryptDataset(
    const FheRuntime& runtime,
    const labml::Dataset& data);

// The runtime selects the sigmoid circuit and depth; the plaintext reference
// must have been trained with the same approximation and optimizer.
EncryptedTrainingResult TrainEncrypted(
    const FheRuntime& runtime,
    const EncryptedDataset& encryptedTrain,
    const labml::Dataset& train,
    const labml::Dataset& test,
    const labml::PlaintextTrainingResult& plaintextReference,
    std::size_t epochs,
    double learningRate,
    RefreshMethod refreshMethod,
    const labml::OptimizerConfiguration& optimizer = {});

}  // namespace labfhe
