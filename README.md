# OpenFHE CKKS Logistic Regression: Simulated and Real Bootstrapping

> Tested with OpenFHE 1.1.2

This project is built from my template project
[`openfhe-template`](https://github.com/joeywwong/openfhe-template) and ports the
original
[`fhe-ckks-lwe-encrypted-ml-lab`](https://github.com/joeywwong/fhe-ckks-lwe-encrypted-ml-lab)
from TenSEAL to C++/OpenFHE.

It preserves the lab's encrypted logistic-regression experiment and compares
two ways of refreshing the encrypted weights and bias:

- **Simulated bootstrapping:** decrypt the model and encrypt it again, matching
  the workaround we used by the lab when TenSEAL did not support CKKS
  bootstrapping.
- **Real bootstrapping:** call OpenFHE's non-interactive `EvalBootstrap` on the
  encrypted weights and encrypted bias once enough natural levels have been
  consumed for OpenFHE to return a genuinely refreshed ciphertext.

Only `LogReg_sample_dataset.csv` and `framingham.csv` are used.

## Lab behavior preserved

- 70% training and 30% test data after a seed-4 shuffle;
- zero-initialized binary logistic regression;
- full-batch gradient descent, 100 epochs, learning rate 0.01;
- cubic sigmoid `0.5 + 0.197z - 0.004z^3` in plaintext and ciphertext training;
- one encrypted feature vector and encrypted label per training record;
- separate encrypted weight and bias ciphertexts;
- per-epoch test accuracy and exact-sigmoid training loss after model decryption;
- the lab's exact Framingham column removal, class balancing, and full-dataset
  standardization order.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the one-to-one mapping and timing
definitions.

## Build and test

Requirements:

- Ubuntu 22.04 or a comparable Linux/WSL environment;
- CMake 3.5.1 or later and a C++17 compiler;
- OpenFHE 1.1.2 installed with its CMake package at `/usr/local/lib/OpenFHE`.

From WSL:

```bash
git clone https://github.com/joeywwong/openfhe-ckks-logistic-regression.git
cd openfhe-ckks-logistic-regression
./scripts/build_and_test_wsl.sh
```

## Run the comparison

The faithful full experiment defaults to 100 epochs for both datasets and both
refresh methods. But it is computationally expensive and takes a long time because it intentionally
retains one ciphertext per sample approach we used in lab, due to Tenseal's limitation. 

A more efficient implementation would pack multiple samples into each CKKS ciphertext and process them in parallel using SIMD batching. This reduces the number of ciphertexts and homomorphic operations.

For the shortest verification run that exercises both simulated and real bootstrapping, use four epochs. The first three epochs consume ciphertext levels, and real bootstrapping is triggered during the fourth epoch:

```bash
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

To run 100 epochs (not recommended, takes a long time)
```bash
./scripts/run_comparison_wsl.sh
```

Direct executable examples:

```bash
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4
./build/openfhe_lab_compare --dataset framingham --refresh both --epochs 4
```

Options:

```text
--dataset logreg|framingham|all
--refresh simulated|real|both
--epochs N
--learning-rate X
--output PATH
```

Per-epoch measurements are written to `results/benchmark.csv`. The interpreted
run is documented in [`docs/RESULTS.md`](docs/RESULTS.md).

## Reported metrics

- encrypted arithmetic time;
- refresh time;
- total seconds per epoch excluding metric calculation;
- metric-only decryption time for the real-bootstrap branch;
- test accuracy;
- exact-sigmoid training loss;
- maximum encrypted-model error versus the matching plaintext epoch;
- paired decrypt+encrypt time on the same worn model whenever a genuine real
  bootstrap occurs;
- consumed CKKS level before and after refresh.

## Repository layout

```text
.
├── app/main.cpp
├── data/
│   ├── LogReg_sample_dataset.csv
│   └── framingham.csv
├── docs/
│   ├── DESIGN.md
│   └── RESULTS.md
├── include/openfhe_lab/
├── results/benchmark.csv
├── scripts/
├── src/
└── tests/plaintext_tests.cpp
```

## Security

The default ring dimension is a laptop demonstration using `HEStd_NotSet`; it
makes no production security claim. Simulated bootstrapping explicitly exposes
the model to the secret-key holder between epochs. See [`SECURITY.md`](SECURITY.md).

## License

MIT. See [`LICENSE`](LICENSE).
