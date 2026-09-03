// Packed matrix-vector reductions and Nesterov updates adapted from
// enc_matrix.h and lr_nag.cpp in
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
using Ciphertext = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

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

void RequirePackedNagState(const FheRuntime& runtime) {
    if (runtime.nagPacking != NagPacking::Packed ||
        !runtime.thetaStateMask || !runtime.phiStateMask) {
        throw std::invalid_argument("Packed NAG state is not configured in this runtime");
    }
}

labml::PlainModel DecryptCombinedModel(
    const FheRuntime& runtime,
    const Ciphertext& model,
    std::size_t featureCount) {
    auto values = DecryptVector(runtime, model, featureCount + 1);
    const double bias = values[featureCount];
    values.resize(featureCount);
    return {std::move(values), bias};
}

Ciphertext EncryptPackedNagState(
    const FheRuntime& runtime,
    const labml::PlainModel& theta,
    const labml::PlainModel& phi) {
    RequirePackedNagState(runtime);
    if (theta.weights.size() != phi.weights.size() ||
        theta.weights.size() + 1 > runtime.rowWidth) {
        throw std::invalid_argument(
            "Complete NAG states must have equal widths that fit in one packed row");
    }

    std::vector<double> packedState(runtime.slots, 0.0);
    for (std::size_t offset = 0; offset < runtime.slots; offset += runtime.rowWidth) {
        const auto& source = ((offset / runtime.rowWidth) % 2 == 0) ? theta : phi;
        std::copy(source.weights.begin(), source.weights.end(), packedState.begin() + offset);
        packedState[offset + source.weights.size()] = source.bias;
    }

    auto plaintext = runtime.context->MakeCKKSPackedPlaintext(
        packedState, 1, 0, nullptr, runtime.slots);
    plaintext->SetLength(runtime.slots);
    return runtime.context->Encrypt(runtime.keyPair.publicKey, plaintext);
}

std::pair<labml::PlainModel, labml::PlainModel> DecryptPackedNagState(
    const FheRuntime& runtime,
    const Ciphertext& packedState,
    std::size_t featureCount) {
    RequirePackedNagState(runtime);
    if (featureCount + 1 > runtime.rowWidth) {
        throw std::invalid_argument("Complete model does not fit in one packed row");
    }
    const std::size_t stateSlots = 2 * runtime.rowWidth;
    const auto values = DecryptVector(runtime, packedState, stateSlots);
    labml::PlainModel theta{
        std::vector<double>(values.begin(), values.begin() + featureCount),
        values[featureCount]};
    labml::PlainModel phi{
        std::vector<double>(
            values.begin() + runtime.rowWidth,
            values.begin() + runtime.rowWidth + featureCount),
        values[runtime.rowWidth + featureCount]};
    return {std::move(theta), std::move(phi)};
}

Ciphertext PackNagState(
    const FheRuntime& runtime,
    const Ciphertext& theta,
    const Ciphertext& phi) {
    RequirePackedNagState(runtime);
    const auto maskedTheta = runtime.context->EvalMult(theta, runtime.thetaStateMask);
    const auto maskedPhi = runtime.context->EvalMult(phi, runtime.phiStateMask);
    return runtime.context->EvalAdd(maskedTheta, maskedPhi);
}

std::pair<Ciphertext, Ciphertext> UnpackNagState(
    const FheRuntime& runtime,
    const Ciphertext& packedState) {
    RequirePackedNagState(runtime);
    const auto maskedTheta = runtime.context->EvalMult(packedState, runtime.thetaStateMask);
    const auto theta = runtime.context->EvalAdd(
        maskedTheta,
        runtime.context->EvalRotate(maskedTheta, static_cast<std::int32_t>(runtime.rowWidth)));
    const auto maskedPhi = runtime.context->EvalMult(packedState, runtime.phiStateMask);
    const auto phi = runtime.context->EvalAdd(
        maskedPhi,
        runtime.context->EvalRotate(maskedPhi, -static_cast<std::int32_t>(runtime.rowWidth)));
    return {theta, phi};
}

Ciphertext SimulatedRefreshPackedNagState(
    const FheRuntime& runtime,
    const Ciphertext& packedState,
    std::size_t featureCount) {
    const auto states = DecryptPackedNagState(runtime, packedState, featureCount);
    return EncryptPackedNagState(runtime, states.first, states.second);
}

EncryptedModel SimulatedRefreshModel(
    const FheRuntime& runtime,
    const EncryptedModel& model,
    std::size_t featureCount) {
    return EncryptModel(runtime, DecryptModel(runtime, model, featureCount));
}

Ciphertext BootstrapCiphertext(const FheRuntime& runtime, const Ciphertext& ciphertext) {
    // Do not count OpenFHE 1.1.2's no-op bootstrap as a refresh.
    if (ciphertext->GetLevel() <= runtime.bootstrapTriggerLevel) {
        return ciphertext;
    }
    auto sparse = ciphertext->Clone();
    sparse->SetSlots(runtime.bootstrapSlots);
    auto refreshed = runtime.context->EvalBootstrap(sparse);
    refreshed->SetSlots(runtime.slots);
    if (refreshed->GetLevel() >= ciphertext->GetLevel()) {
        throw std::runtime_error("Bootstrapping did not restore optimizer-state levels");
    }
    return refreshed;
}

EncryptedModel BootstrapModel(const FheRuntime& runtime, const EncryptedModel& model) {
    return {
        BootstrapCiphertext(runtime, model.weights),
        BootstrapCiphertext(runtime, model.bias),
    };
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvaluatePolynomialSigmoid(
    const FheRuntime& runtime,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& score) {
    if (runtime.sigmoid == labml::SigmoidApproximation::Cubic) {
        // Preserve the lab/main-branch polynomial and its evaluation circuit.
        const auto squared = runtime.context->EvalMult(score, score);
        const auto cubed   = runtime.context->EvalMult(squared, score);
        const auto linear  = runtime.context->EvalMult(score, 0.197);
        const auto cubic   = runtime.context->EvalMult(cubed, -0.004);
        return runtime.context->EvalAdd(runtime.context->EvalAdd(linear, cubic), 0.5);
    }
    return runtime.context->EvalLogistic(
        score,
        labml::kSigmoidApproximationLowerBound,
        labml::kSigmoidApproximationUpperBound,
        labml::kSigmoidApproximationDegree);
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

Ciphertext TrainOnePackedNagEpoch(
    const FheRuntime& runtime,
    const EncryptedDataset& encryptedTrain,
    const Ciphertext& model,
    double learningRate) {
    Ciphertext gradientSum;
    bool hasGradient = false;

    for (const auto& block : encryptedTrain.blocks) {
        // Packed NAG appends a public intercept value of one to each real
        // feature row. The combined model row is [weights, bias, padding].
        const auto coordinateProducts = runtime.context->EvalMult(block.features, model);
        const auto score = runtime.context->EvalSumCols(
            coordinateProducts, runtime.rowWidth, *runtime.sumColsKeys);
        const auto output = EvaluatePolynomialSigmoid(runtime, score);
        const auto error = runtime.context->EvalSub(output, block.labels);
        const auto coordinateGradients = runtime.context->EvalMult(block.features, error);
        const auto blockGradient = runtime.context->EvalSumRows(
            coordinateGradients, runtime.rowWidth, *runtime.sumRowsKeys);
        gradientSum = hasGradient
            ? runtime.context->EvalAdd(gradientSum, blockGradient)
            : blockGradient;
        hasGradient = true;
    }

    const double step = learningRate / static_cast<double>(encryptedTrain.sampleCount);
    return runtime.context->EvalSub(
        model, runtime.context->EvalMult(gradientSum, step));
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

std::string NagPackingName(NagPacking packing) {
    switch (packing) {
        case NagPacking::Separate:
            return "separate";
        case NagPacking::Packed:
            return "packed";
    }
    throw std::invalid_argument("Unknown NAG packing");
}

FheRuntime CreateFheRuntime(const CkksConfiguration& configuration) {
    labml::SigmoidApproximationName(configuration.sigmoid);
    NagPackingName(configuration.nagPacking);
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
    if (configuration.nagPacking == NagPacking::Packed &&
        configuration.bootstrapSlots < 2 * configuration.rowWidth) {
        throw std::invalid_argument(
            "Packed NAG needs at least two rows of bootstrap slots");
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
    const std::uint32_t defaultLevelsAvailableAfterBootstrap =
        (configuration.sigmoid == labml::SigmoidApproximation::Chebyshev ? 16U : 10U) +
        (configuration.nagPacking == NagPacking::Packed ? 2U : 0U);
    const std::uint32_t levelsAvailableAfterBootstrap =
        configuration.levelsAvailableAfterBootstrap != 0
            ? configuration.levelsAvailableAfterBootstrap
            : defaultLevelsAvailableAfterBootstrap;
    const std::uint32_t multiplicativeDepth =
        bootstrapDepth + levelsAvailableAfterBootstrap;
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
    lbcrypto::Plaintext thetaStateMask;
    lbcrypto::Plaintext phiStateMask;
    if (configuration.nagPacking == NagPacking::Packed) {
        const auto rotation = static_cast<std::int32_t>(configuration.rowWidth);
        context->EvalRotateKeyGen(keyPair.secretKey, {rotation, -rotation});
        std::vector<double> thetaMask(configuration.slots, 0.0);
        std::vector<double> phiMask(configuration.slots, 0.0);
        for (std::size_t offset = 0; offset < configuration.slots;
             offset += configuration.rowWidth) {
            auto& mask = ((offset / configuration.rowWidth) % 2 == 0)
                ? thetaMask : phiMask;
            std::fill(
                mask.begin() + offset,
                mask.begin() + offset + configuration.rowWidth,
                1.0);
        }
        thetaStateMask = context->MakeCKKSPackedPlaintext(
            thetaMask, 1, 0, nullptr, configuration.slots);
        phiStateMask = context->MakeCKKSPackedPlaintext(
            phiMask, 1, 0, nullptr, configuration.slots);
        thetaStateMask->SetLength(configuration.slots);
        phiStateMask->SetLength(configuration.slots);
    }
    context->EvalBootstrapKeyGen(keyPair.secretKey, configuration.bootstrapSlots);
    const std::uint32_t bootstrapTriggerLevel =
        multiplicativeDepth - levelsAvailableAfterBootstrap;
    return {context, keyPair, multiplicativeDepth, configuration.slots,
            configuration.rowWidth, configuration.bootstrapSlots,
            sumRowsKeys, sumColsKeys, thetaStateMask, phiStateMask,
            bootstrapTriggerLevel, configuration.sigmoid, configuration.nagPacking};
}

EncryptedDataset EncryptDataset(const FheRuntime& runtime, const labml::Dataset& data) {
    EncryptedDataset result;
    result.sampleCount = data.size();
    result.featureCount = labml::FeatureCount(data);
    if (runtime.nagPacking == NagPacking::Packed &&
        result.featureCount + 1 > runtime.rowWidth) {
        throw std::invalid_argument(
            "Packed NAG row width must include all features and one intercept");
    }
    auto packedBlocks = PackTrainingData(data, runtime.slots, runtime.rowWidth);
    if (runtime.nagPacking == NagPacking::Packed) {
        for (auto& block : packedBlocks) {
            for (std::size_t row = 0; row < block.sampleCount; ++row) {
                block.features[row * runtime.rowWidth + result.featureCount] = 1.0;
            }
        }
    }
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
    RefreshMethod refreshMethod,
    const labml::OptimizerConfiguration& optimizer) {
    labml::ValidateOptimizerConfiguration(optimizer);
    labml::SigmoidApproximationName(runtime.sigmoid);
    if (plaintextReference.sigmoid != runtime.sigmoid) {
        throw std::invalid_argument("Plaintext reference must use the same sigmoid approximation");
    }
    RefreshMethodName(refreshMethod);
    if (!std::isfinite(learningRate) || learningRate <= 0.0 || test.empty()) {
        throw std::invalid_argument("Learning rate must be positive and finite; test data must not be empty");
    }
    if (encryptedTrain.blocks.empty() || encryptedTrain.sampleCount != train.size() ||
        encryptedTrain.featureCount != labml::FeatureCount(train)) {
        throw std::invalid_argument("Encrypted and plaintext training sets must be non-empty and aligned");
    }
    if (epochs == 0 || plaintextReference.epochs.size() != epochs) {
        throw std::invalid_argument("Plaintext reference must contain one record per requested epoch");
    }

    if (plaintextReference.optimizer.method != optimizer.method ||
        (optimizer.method == labml::Optimizer::NesterovAcceleratedGradient &&
         plaintextReference.optimizer.momentum != optimizer.momentum)) {
        throw std::invalid_argument("Plaintext reference must use the same optimizer and momentum");
    }

    const bool isNesterov =
        optimizer.method == labml::Optimizer::NesterovAcceleratedGradient;
    const bool useMomentum = isNesterov && optimizer.momentum > 0.0;
    if (runtime.nagPacking == NagPacking::Packed && !isNesterov) {
        throw std::invalid_argument("Packed NAG requires the Nesterov optimizer");
    }
    const bool packNesterovState =
        isNesterov && runtime.nagPacking == NagPacking::Packed;
    const std::size_t featureCount = labml::FeatureCount(train);
    labml::PlainModel currentPlainModel{std::vector<double>(featureCount, 0.0), 0.0};
    EncryptedModel encryptedModel;
    // Preserve the previous unaccelerated step, not the previous look-ahead.
    EncryptedModel previousStep;
    Ciphertext combinedModel;
    Ciphertext previousCombinedStep;
    Ciphertext packedNesterovState;
    if (packNesterovState) {
        packedNesterovState =
            EncryptPackedNagState(runtime, currentPlainModel, currentPlainModel);
    }
    else {
        encryptedModel = EncryptModel(runtime, currentPlainModel);
    }

    EncryptedTrainingResult result;
    result.epochs.reserve(epochs);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "    epoch " << (epoch + 1) << '/' << epochs << ": encrypted training" << std::flush;
        const auto homomorphicStart = Clock::now();
        EncryptedModel updatedModel;
        Ciphertext updatedCombinedModel;
        if (packNesterovState) {
            auto states = UnpackNagState(runtime, packedNesterovState);
            combinedModel = std::move(states.first);
            previousCombinedStep = std::move(states.second);
            const auto gradientStep =
                TrainOnePackedNagEpoch(runtime, encryptedTrain, combinedModel, learningRate);
            updatedCombinedModel = gradientStep;
            if (useMomentum && epoch > 0) {
                updatedCombinedModel = runtime.context->EvalAdd(
                    gradientStep,
                    runtime.context->EvalMult(
                        runtime.context->EvalSub(gradientStep, previousCombinedStep),
                        optimizer.momentum));
            }
            previousCombinedStep = gradientStep;
            packedNesterovState =
                PackNagState(runtime, updatedCombinedModel, previousCombinedStep);
        }
        else {
            const auto gradientStep =
                TrainOneEpoch(runtime, encryptedTrain, encryptedModel, learningRate);
            updatedModel = gradientStep;
            if (useMomentum) {
                if (epoch > 0) {
                    updatedModel.weights = runtime.context->EvalAdd(
                        gradientStep.weights, runtime.context->EvalMult(
                            runtime.context->EvalSub(gradientStep.weights, previousStep.weights),
                            optimizer.momentum));
                    updatedModel.bias = runtime.context->EvalAdd(
                        gradientStep.bias, runtime.context->EvalMult(
                            runtime.context->EvalSub(gradientStep.bias, previousStep.bias),
                            optimizer.momentum));
                }
                previousStep = gradientStep;
            }
        }
        const auto homomorphicEnd = Clock::now();
        const std::uint32_t levelBefore = packNesterovState
            ? packedNesterovState->GetLevel()
            : (useMomentum
                ? std::max(ConsumedLevel(updatedModel), ConsumedLevel(previousStep))
                : ConsumedLevel(updatedModel));

        double refreshSeconds = 0.0;
        double pairedSimulatedRefreshSeconds = 0.0;
        double metricDecryptionSeconds = 0.0;
        bool refreshed = false;
        if (refreshMethod == RefreshMethod::SimulatedBootstrapping) {
            std::cout << "; decrypt + encrypt" << std::flush;
            const auto refreshStart = Clock::now();
            if (packNesterovState) {
                const auto states =
                    DecryptPackedNagState(runtime, packedNesterovState, featureCount);
                currentPlainModel = states.first;
                packedNesterovState =
                    EncryptPackedNagState(runtime, states.first, states.second);
            }
            else {
                currentPlainModel = DecryptModel(runtime, updatedModel, featureCount);
                encryptedModel = EncryptModel(runtime, currentPlainModel);
                if (useMomentum) {
                    previousStep = SimulatedRefreshModel(runtime, previousStep, featureCount);
                }
            }
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
            if (packNesterovState) {
                const auto comparisonPackedState = SimulatedRefreshPackedNagState(
                    runtime, packedNesterovState, featureCount);
                static_cast<void>(comparisonPackedState);
            }
            else {
                const auto comparisonEncryptedModel =
                    SimulatedRefreshModel(runtime, updatedModel, featureCount);
                static_cast<void>(comparisonEncryptedModel);
                if (useMomentum) {
                    const auto comparisonPreviousStep =
                        SimulatedRefreshModel(runtime, previousStep, featureCount);
                    static_cast<void>(comparisonPreviousStep);
                }
            }
            const auto simulatedEnd = Clock::now();
            pairedSimulatedRefreshSeconds = SecondsBetween(simulatedStart, simulatedEnd);

            const auto refreshStart = Clock::now();
            // As in the official example, only the periodic model is sparsely
            // bootstrapped. The matrix inputs stay fully packed and unchanged.
            if (packNesterovState) {
                packedNesterovState =
                    BootstrapCiphertext(runtime, packedNesterovState);
            }
            else {
                encryptedModel = BootstrapModel(runtime, updatedModel);
                if (useMomentum) {
                    previousStep = BootstrapModel(runtime, previousStep);
                }
            }
            const auto refreshEnd = Clock::now();
            refreshSeconds = SecondsBetween(refreshStart, refreshEnd);
            refreshed = true;

            // The lab decrypted after every epoch to report accuracy and loss.
            // This copy is only for metrics and does not replace the bootstrapped
            // ciphertext used by the next epoch.
            const auto metricDecryptStart = Clock::now();
            currentPlainModel = packNesterovState
                ? DecryptPackedNagState(runtime, packedNesterovState, featureCount).first
                : DecryptModel(runtime, encryptedModel, featureCount);
            const auto metricDecryptEnd = Clock::now();
            metricDecryptionSeconds = SecondsBetween(metricDecryptStart, metricDecryptEnd);
        }
        else {
            std::cout << "; bootstrap deferred" << std::flush;
            if (!packNesterovState) {
                encryptedModel = updatedModel;
            }
            const auto metricDecryptStart = Clock::now();
            currentPlainModel = packNesterovState
                ? DecryptCombinedModel(runtime, updatedCombinedModel, featureCount)
                : DecryptModel(runtime, updatedModel, featureCount);
            const auto metricDecryptEnd = Clock::now();
            metricDecryptionSeconds = SecondsBetween(metricDecryptStart, metricDecryptEnd);
        }

        const std::uint32_t levelAfter = packNesterovState
            ? packedNesterovState->GetLevel()
            : (useMomentum
                ? std::max(ConsumedLevel(encryptedModel), ConsumedLevel(previousStep))
                : ConsumedLevel(encryptedModel));
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
