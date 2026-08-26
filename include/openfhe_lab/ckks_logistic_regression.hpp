#pragma once

#include "openfhe_lab/dataset.hpp"
#include "openfhe_lab/logistic_regression.hpp"

#include "openfhe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace labfhe {

enum class RefreshMethod {
    SimulatedBootstrapping,
    RealBootstrapping,
};

std::string RefreshMethodName(RefreshMethod method);

struct CkksConfiguration {
    // Nine Framingham features require at least nine active slots. Sixteen is
    // the next sparse power-of-two slot count supported by the bootstrap setup.
    std::uint32_t slots{16};
    std::vector<std::uint32_t> levelBudget{3, 3};
    std::uint32_t levelsAvailableAfterBootstrap{10};
};

struct FheRuntime {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    std::uint32_t multiplicativeDepth{};
    std::uint32_t slots{};
    // EvalBootstrap in OpenFHE 1.1.2 returns the original ciphertext when it
    // still has at least as many towers as bootstrapping would return. A model
    // must be beyond this consumed level for refresh to be genuine.
    std::uint32_t bootstrapTriggerLevel{};
};

struct EncryptedSample {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> features;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> label;
};

using EncryptedDataset = std::vector<EncryptedSample>;

struct EncryptedModel {
    // The lab encrypted weights and bias separately. This project deliberately
    // keeps that layout so refresh means two ciphertext refresh operations.
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

EncryptedTrainingResult TrainEncrypted(
    const FheRuntime& runtime,
    const EncryptedDataset& encryptedTrain,
    const labml::Dataset& train,
    const labml::Dataset& test,
    const labml::PlaintextTrainingResult& plaintextReference,
    std::size_t epochs,
    double learningRate,
    RefreshMethod refreshMethod);

}  // namespace labfhe
