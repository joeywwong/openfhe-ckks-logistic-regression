#include "openfhe_lab/dataset.hpp"
#include "openfhe_lab/logistic_regression.hpp"

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

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Require(argc == 3, "Expected LogReg and Framingham CSV paths");
        Require(std::abs(labml::PolynomialSigmoid(0.0) - 0.5) < 1e-12,
                "Polynomial sigmoid must map zero to 0.5");
        Require(std::abs(labml::ExactSigmoid(0.0) - 0.5) < 1e-12,
                "Exact sigmoid must map zero to 0.5");

        const auto logreg = labml::LoadLogRegSample(argv[1]);
        Require(logreg.size() == 1000, "LogReg sample must contain 1000 records");
        Require(labml::FeatureCount(logreg) == 2, "LogReg sample must contain two features");
        const auto logregSplit = labml::LabTrainTestSplit(logreg, 0.30, 4);
        Require(logregSplit.train.size() == 700 && logregSplit.test.size() == 300,
                "LogReg split must reproduce the lab's 70/30 sizes");

        const auto framingham = labml::LoadAndPrepareFramingham(argv[2]);
        Require(framingham.size() == 1114, "Prepared Framingham data must contain 1114 records");
        Require(labml::FeatureCount(framingham) == 9, "Prepared Framingham data must contain nine features");
        const auto framinghamSplit = labml::LabTrainTestSplit(framingham, 0.30, 4);
        Require(framinghamSplit.train.size() == 780 && framinghamSplit.test.size() == 334,
                "Framingham split must reproduce the lab's 70/30 sizes");

        const auto trained = labml::TrainPlaintext(
            logregSplit.train, logregSplit.test, 2, 0.01);
        Require(trained.epochs.size() == 2, "Plaintext trainer must report every epoch");
        Require(std::isfinite(trained.epochs.back().loss), "Plaintext loss must be finite");
        Require(trained.epochs.back().accuracy >= 0.95,
                "Linearly separable LogReg sample should classify accurately");

        std::cout << "All plaintext tests passed\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
