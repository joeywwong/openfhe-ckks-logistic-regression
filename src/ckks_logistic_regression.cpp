// Packed matrix-vector reductions adapted from enc_matrix.h in
// openfheorg/openfhe-logreg-training-examples (BSD-2-Clause).
// Copyright (c) 2023, Duality Technologies Inc. All rights reserved.
// See THIRD_PARTY_NOTICES.md for the retained upstream license.
#include "openfhe_lab/ckks_logistic_regression.hpp"
#include "openfhe_lab/sample_packing.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace labfhe {
namespace {

using Clock = std::chrono::steady_clock;

double SecondsBetween(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

std::vector<double> DecryptVector(
    const FheRuntime& runtime,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ciphertext,
    std::size_t outputLength) {
    lbcrypto::Plaintext plaintext;
    const auto result = runtime.context->Decrypt(runtime.keyPair.secretKey, ciphertext, &plaintext);
    if (!result.isValid) {
        throw std::runtime_error("OpenFHE decryption failed");
    }
    plaintext->SetLength(outputLength);
    auto values = plaintext->GetRealPackedValue();
    values.resize(outputLength);
    return values;
}

EncryptedModel EncryptModel(const FheRuntime& runtime, const labml::PlainModel& model) {
    if (model.weights.size() > runtime.rowWidth) {
        throw std::invalid_argument("Model weights do not fit in the packed row width");
    }
    std::vector<double> packedWeights(runtime.slots, 0.0);
    // Row-cloned weights: [w0,w1,...,padding] repeated once per sample row.
    for (std::size_t offset = 0; offset < runtime.slots; offset += runtime.rowWidth) {
        std::copy(model.weights.begin(), model.weights.end(), packedWeights.begin() + offset);
    }
    const std::vector<double> packedBias(runtime.slots, model.bias);

    auto weightsPlaintext = runtime.context->MakeCKKSPackedPlaintext(
        packedWeights, 1, 0, nullptr, runtime.slots);
    auto biasPlaintext = runtime.context->MakeCKKSPackedPlaintext(
        packedBias, 1, 0, nullptr, runtime.slots);
    weightsPlaintext->SetLength(runtime.slots);
    biasPlaintext->SetLength(runtime.slots);
    return {
        runtime.context->Encrypt(runtime.keyPair.publicKey, weightsPlaintext),
        runtime.context->Encrypt(runtime.keyPair.publicKey, biasPlaintext),
    };
}

labml::PlainModel DecryptModel(
    const FheRuntime& runtime,
    const EncryptedModel& model,
    std::size_t featureCount) {
    auto weights = DecryptVector(runtime, model.weights, featureCount);
    const double bias = DecryptVector(runtime, model.bias, 1).front();
    return {std::move(weights), bias};
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvaluatePolynomialSigmoid(
    const FheRuntime& runtime,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& score) {
    const auto squared = runtime.context->EvalMult(score, score);
    const auto cubed   = runtime.context->EvalMult(squared, score);
    const auto linear  = runtime.context->EvalMult(score, 0.197);
    const auto cubic   = runtime.context->EvalMult(cubed, -0.004);
    return runtime.context->EvalAdd(runtime.context->EvalAdd(linear, cubic), 0.5);
}

EncryptedModel TrainOneEpoch(
    const FheRuntime& runtime,
    const EncryptedDataset& encryptedTrain,
    const EncryptedModel& model,
    double learningRate) {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> weightGradientSum;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> biasGradientSum;
    bool hasGradient = false;

    for (const auto& block : encryptedTrain.blocks) {
        const auto coordinateProducts = runtime.context->EvalMult(block.features, model.weights);
        // Official MatrixVectorProductRow: sum feature columns independently
        // for every row and replicate the dot product across that row.
        const auto dotProduct = runtime.context->EvalSumCols(
            coordinateProducts, runtime.rowWidth, *runtime.sumColsKeys);
        const auto score = runtime.context->EvalAdd(dotProduct, model.bias);
        const auto output = EvaluatePolynomialSigmoid(runtime, score);
        const auto error = runtime.context->EvalSub(output, block.labels);
        const auto weightProducts = runtime.context->EvalMult(block.features, error);
        // Official MatrixVectorProductCol: reduce sample rows by feature.
        // The result is a row-cloned gradient, matching the weights' layout.
        const auto blockWeightGradient = runtime.context->EvalSumRows(
            weightProducts, runtime.rowWidth, *runtime.sumRowsKeys);
        // Zero-padded X rows have zero weight gradients, but sigmoid(bias) is
        // NOT zero. Exclude padded rows explicitly from the bias gradient.
        const auto validErrors = runtime.context->EvalMult(error, block.validRows);
        const auto blockBiasGradient = runtime.context->EvalSumRows(
            validErrors, runtime.rowWidth, *runtime.sumRowsKeys);

        if (!hasGradient) {
            weightGradientSum = blockWeightGradient;
            biasGradientSum   = blockBiasGradient;
            hasGradient       = true;
        }
        else {
            weightGradientSum = runtime.context->EvalAdd(weightGradientSum, blockWeightGradient);
            biasGradientSum   = runtime.context->EvalAdd(biasGradientSum, blockBiasGradient);
        }
    }

    // All blocks contribute before ONE update: this is still full-batch GD.
    // Divide by actual records, never by the padded row count.
    const double step = learningRate / static_cast<double>(encryptedTrain.sampleCount);
    const auto weightUpdate = runtime.context->EvalMult(weightGradientSum, step);
    const auto biasUpdate   = runtime.context->EvalMult(biasGradientSum, step);
    return {
        runtime.context->EvalSub(model.weights, weightUpdate),
        runtime.context->EvalSub(model.bias, biasUpdate),
    };
}

std::uint32_t ConsumedLevel(const EncryptedModel& model) {
    return std::max(model.weights->GetLevel(), model.bias->GetLevel());
}

}  // namespace

std::string RefreshMethodName(RefreshMethod method) {
    switch (method) {
        case RefreshMethod::SimulatedBootstrapping:
            return "simulated_bootstrapping";
        case RefreshMethod::RealBootstrapping:
            return "real_bootstrapping";
    }
    throw std::invalid_argument("Unknown refresh method");
}

FheRuntime CreateFheRuntime(const CkksConfiguration& configuration) {
    if (configuration.slots != 2048) {
        throw std::invalid_argument("Matrix reductions require all 2048 slots in the 4096-degree demo ring");
    }
    if (configuration.rowWidth == 0 ||
        (configuration.rowWidth & (configuration.rowWidth - 1)) != 0 ||
        configuration.bootstrapSlots == 0 ||
        (configuration.bootstrapSlots & (configuration.bootstrapSlots - 1)) != 0 ||
        configuration.rowWidth > configuration.bootstrapSlots ||
        configuration.bootstrapSlots > configuration.slots) {
        throw std::invalid_argument("Row width and bootstrap slots must be compatible powers of two");
    }
    if (configuration.levelBudget.size() != 2) {
        throw std::invalid_argument("CKKS bootstrapping needs a two-entry level budget");
    }

    constexpr auto secretKeyDistribution = lbcrypto::UNIFORM_TERNARY;
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDistribution);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetRingDim(1U << 12U);
    parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    parameters.SetNumLargeDigits(3);
    parameters.SetBatchSize(configuration.slots);
    parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    const std::uint32_t bootstrapDepth = lbcrypto::FHECKKSRNS::GetBootstrapDepth(
        configuration.levelBudget, secretKeyDistribution);
    const std::uint32_t multiplicativeDepth =
        bootstrapDepth + configuration.levelsAvailableAfterBootstrap;
    parameters.SetMultiplicativeDepth(multiplicativeDepth);

    auto context = lbcrypto::GenCryptoContext(parameters);
    context->Enable(lbcrypto::PKE);
    context->Enable(lbcrypto::KEYSWITCH);
    context->Enable(lbcrypto::LEVELEDSHE);
    context->Enable(lbcrypto::ADVANCEDSHE);
    context->Enable(lbcrypto::FHE);

    const std::vector<std::uint32_t> bsgsDimensions{0, 0};
    context->EvalBootstrapSetup(configuration.levelBudget, bsgsDimensions, configuration.bootstrapSlots);

    auto keyPair = context->KeyGen();
    if (!keyPair.good()) {
        throw std::runtime_error("OpenFHE key generation failed");
    }
    context->EvalMultKeyGen(keyPair.secretKey);
    context->EvalSumKeyGen(keyPair.secretKey);
    auto sumRowsKeys = context->EvalSumRowsKeyGen(keyPair.secretKey, nullptr, configuration.rowWidth);
    auto sumColsKeys = context->EvalSumColsKeyGen(keyPair.secretKey);
    context->EvalBootstrapKeyGen(keyPair.secretKey, configuration.bootstrapSlots);
    const std::uint32_t bootstrapTriggerLevel =
        multiplicativeDepth - configuration.levelsAvailableAfterBootstrap;
    return {context, keyPair, multiplicativeDepth, configuration.slots,
            configuration.rowWidth, configuration.bootstrapSlots,
            sumRowsKeys, sumColsKeys, bootstrapTriggerLevel};
}

EncryptedDataset EncryptDataset(const FheRuntime& runtime, const labml::Dataset& data) {
    EncryptedDataset result;
    const auto packedBlocks = PackTrainingData(data, runtime.slots, runtime.rowWidth);
    result.sampleCount = data.size();
    result.featureCount = labml::FeatureCount(data);
    result.blocks.reserve(packedBlocks.size());
    for (const auto& block : packedBlocks) {
        auto featurePlaintext = runtime.context->MakeCKKSPackedPlaintext(
            block.features, 1, 0, nullptr, runtime.slots);
        auto labelPlaintext = runtime.context->MakeCKKSPackedPlaintext(
            block.labels, 1, 0, nullptr, runtime.slots);
        auto validRows = runtime.context->MakeCKKSPackedPlaintext(
            block.validRows, 1, 0, nullptr, runtime.slots);
        featurePlaintext->SetLength(runtime.slots);
        labelPlaintext->SetLength(runtime.slots);
        result.blocks.push_back({
            runtime.context->Encrypt(runtime.keyPair.publicKey, featurePlaintext),
            runtime.context->Encrypt(runtime.keyPair.publicKey, labelPlaintext),
            validRows,
            block.sampleCount,
        });
    }
    return result;
}

EncryptedTrainingResult TrainEncrypted(
    const FheRuntime& runtime,
    const EncryptedDataset& encryptedTrain,
    const labml::Dataset& train,
    const labml::Dataset& test,
    const labml::PlaintextTrainingResult& plaintextReference,
    std::size_t epochs,
    double learningRate,
    RefreshMethod refreshMethod) {
    if (encryptedTrain.blocks.empty() || encryptedTrain.sampleCount != train.size() ||
        encryptedTrain.featureCount != labml::FeatureCount(train)) {
        throw std::invalid_argument("Encrypted and plaintext training sets must be non-empty and aligned");
    }
    if (epochs == 0 || plaintextReference.epochs.size() != epochs) {
        throw std::invalid_argument("Plaintext reference must contain one record per requested epoch");
    }

    const std::size_t featureCount = labml::FeatureCount(train);
    labml::PlainModel currentPlainModel{std::vector<double>(featureCount, 0.0), 0.0};
    auto encryptedModel = EncryptModel(runtime, currentPlainModel);

    EncryptedTrainingResult result;
    result.epochs.reserve(epochs);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "    epoch " << (epoch + 1) << '/' << epochs << ": encrypted training" << std::flush;
        const auto homomorphicStart = Clock::now();
        auto updatedModel = TrainOneEpoch(runtime, encryptedTrain, encryptedModel, learningRate);
        const auto homomorphicEnd = Clock::now();
        const std::uint32_t levelBefore = ConsumedLevel(updatedModel);

        double refreshSeconds = 0.0;
        double pairedSimulatedRefreshSeconds = 0.0;
        double metricDecryptionSeconds = 0.0;
        bool refreshed = false;
        if (refreshMethod == RefreshMethod::SimulatedBootstrapping) {
            std::cout << "; decrypt + encrypt" << std::flush;
            const auto refreshStart = Clock::now();
            currentPlainModel = DecryptModel(runtime, updatedModel, featureCount);
            encryptedModel = EncryptModel(runtime, currentPlainModel);
            const auto refreshEnd = Clock::now();
            refreshSeconds = SecondsBetween(refreshStart, refreshEnd);
            pairedSimulatedRefreshSeconds = refreshSeconds;
            refreshed = true;
        }
        else if (levelBefore > runtime.bootstrapTriggerLevel) {
            std::cout << "; EvalBootstrap" << std::flush;

            // Time the lab workaround on the exact same worn ciphertexts, but
            // discard its refreshed result. This is a paired refresh benchmark;
            // the real-mode model below still comes only from EvalBootstrap.
            const auto simulatedStart = Clock::now();
            const auto comparisonPlainModel = DecryptModel(runtime, updatedModel, featureCount);
            const auto comparisonEncryptedModel = EncryptModel(runtime, comparisonPlainModel);
            static_cast<void>(comparisonEncryptedModel);
            const auto simulatedEnd = Clock::now();
            pairedSimulatedRefreshSeconds = SecondsBetween(simulatedStart, simulatedEnd);

            const auto refreshStart = Clock::now();
            // As in the official example, only the periodic model is sparsely
            // bootstrapped. The matrix inputs stay fully packed and unchanged.
            auto sparseWeights = updatedModel.weights->Clone();
            auto sparseBias = updatedModel.bias->Clone();
            sparseWeights->SetSlots(runtime.bootstrapSlots);
            sparseBias->SetSlots(runtime.bootstrapSlots);
            encryptedModel.weights = runtime.context->EvalBootstrap(sparseWeights);
            encryptedModel.bias    = runtime.context->EvalBootstrap(sparseBias);
            encryptedModel.weights->SetSlots(runtime.slots);
            encryptedModel.bias->SetSlots(runtime.slots);
            const auto refreshEnd = Clock::now();
            refreshSeconds = SecondsBetween(refreshStart, refreshEnd);
            refreshed = true;

            // The lab decrypted after every epoch to report accuracy and loss.
            // This copy is only for metrics and does not replace the bootstrapped
            // ciphertext used by the next epoch.
            const auto metricDecryptStart = Clock::now();
            currentPlainModel = DecryptModel(runtime, encryptedModel, featureCount);
            const auto metricDecryptEnd = Clock::now();
            metricDecryptionSeconds = SecondsBetween(metricDecryptStart, metricDecryptEnd);
        }
        else {
            std::cout << "; bootstrap deferred" << std::flush;
            encryptedModel = updatedModel;
            const auto metricDecryptStart = Clock::now();
            currentPlainModel = DecryptModel(runtime, encryptedModel, featureCount);
            const auto metricDecryptEnd = Clock::now();
            metricDecryptionSeconds = SecondsBetween(metricDecryptStart, metricDecryptEnd);
        }

        const std::uint32_t levelAfter = ConsumedLevel(encryptedModel);
        const double homomorphicSeconds = SecondsBetween(homomorphicStart, homomorphicEnd);
        const double accuracy = labml::Accuracy(currentPlainModel, test);
        const double loss = labml::ExactLogLoss(currentPlainModel, train);
        const double modelError = labml::MaximumModelError(
            currentPlainModel, plaintextReference.epochs[epoch].model);

        result.epochs.push_back({
            epoch + 1,
            homomorphicSeconds,
            refreshSeconds,
            pairedSimulatedRefreshSeconds,
            metricDecryptionSeconds,
            homomorphicSeconds + refreshSeconds,
            accuracy,
            loss,
            modelError,
            refreshed,
            levelBefore,
            levelAfter,
        });

        std::cout << std::fixed << std::setprecision(3)
                  << " done; epoch " << (homomorphicSeconds + refreshSeconds) << " s, refresh "
                  << refreshSeconds << " s";
        if (refreshed && refreshMethod == RefreshMethod::RealBootstrapping) {
            std::cout << " (paired decrypt+encrypt " << pairedSimulatedRefreshSeconds << " s)";
        }
        std::cout << std::setprecision(6) << ", accuracy " << accuracy << ", loss " << loss
                  << ", consumed level " << levelBefore << " -> " << levelAfter << '\n';
    }
    result.finalModel = currentPlainModel;
    return result;
}

}  // namespace labfhe
