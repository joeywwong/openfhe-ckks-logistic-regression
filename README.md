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
  the workaround used in the lab when TenSEAL did not support CKKS
  bootstrapping.
- **Real bootstrapping:** call OpenFHE's non-interactive `EvalBootstrap` on the
  encrypted weights and encrypted bias once enough natural levels have been
  consumed for OpenFHE to return a genuinely refreshed ciphertext.

Only `LogReg_sample_dataset.csv` and `framingham.csv` are used.

Training samples now use **row-major CKKS SIMD packing**, adapted from the
[official OpenFHE logistic-regression example](https://github.com/openfheorg/openfhe-logreg-training-examples).
This requested optimization changes the ciphertext layout, not the lab's
data processing, model, optimizer, or sigmoid. The official example's Nesterov
momentum and Chebyshev sigmoid are not adopted here.

## Lab behavior preserved

- 70% training and 30% test data after a seed-4 shuffle;
- zero-initialized binary logistic regression;
- full-batch gradient descent, 100 epochs, learning rate 0.01;
- cubic sigmoid `0.5 + 0.197z - 0.004z^3` in plaintext and ciphertext training;
- encrypted features and labels, now batched into row-major ciphertext blocks;
- separate encrypted weight and bias ciphertexts;
- per-epoch test accuracy and exact-sigmoid training loss after model decryption;
- the lab's exact Framingham column removal, class balancing, and full-dataset
  standardization order.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the one-to-one mapping and timing
definitions.

## Sample packing

Each ciphertext contains sample rows padded to a power-of-two feature width.
Weights repeat across rows and labels repeat across columns. `EvalSumCols`
computes row-wise scores and `EvalSumRows` aggregates gradients across samples.
All blocks contribute to one full-batch update; padded rows are masked out of
the bias gradient.

| Dataset | Training rows | Row width | Rows/block | Blocks | Input ciphertexts, before -> after |
|---|---:|---:|---:|---:|---:|
| LogReg sample | 700 | 2 | 1,024 | 1 | 1,400 -> 2 |
| Framingham | 780 | 16 | 128 | 7 | 1,560 -> 14 |

The existing ring dimension remains 4,096. Data use all 2,048 slots; the
repeated model still uses 16-slot sparse bootstrapping, with weights and bias
refreshed separately. See [the design](docs/DESIGN.md#packed-ciphertext-layout)
and [four-epoch packed results](docs/PACKED_RESULTS.md).

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

Tests include plaintext checks, packing/padding checks, and encrypted tests on
subsets of the two lab datasets, including training after a real bootstrap.
No CMake presets are needed. The workflow was verified with CMake 3.22.1;
the CMake 3.5.1 compatibility branch has not been executed locally.

## Run the comparison

The experiment retains the lab default of 100 epochs for both datasets and both
refresh methods. Packing reduces the number of encrypted operations. For a
four-epoch verification run covering both refresh methods:

```bash
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

In the measured packed run, genuine real bootstrapping occurred in epochs 3
and 4. The extra packed reductions change level consumption compared with
the old per-sample implementation; refresh is still triggered by consumed
levels, not by a fixed epoch number.

To run the lab's full 100 epochs (not represented as measured by the four-epoch report):

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

New measurements go to `results/benchmark_packed.csv`. `OUTPUT_PATH` overrides
that path when using the script. The earlier `results/benchmark.csv` is left
intact. See [`docs/PACKED_RESULTS.md`](docs/PACKED_RESULTS.md) for the new run;
[`docs/RESULTS.md`](docs/RESULTS.md) is the historical unpacked report.

## Reported metrics

- encrypted arithmetic time;
- refresh time;
- training seconds per epoch (arithmetic plus refresh), excluding metrics and
  the discarded paired-refresh measurement;
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
│   ├── RESULTS.md
│   └── PACKED_RESULTS.md
├── include/openfhe_lab/
├── results/
│   ├── benchmark.csv
│   └── benchmark_packed.csv
├── scripts/
├── src/
└── tests/
```

## Security

The default ring dimension is a laptop demonstration using `HEStd_NotSet`; it
makes no production security claim. Simulated bootstrapping explicitly exposes
the model to the secret-key holder between epochs. See [`SECURITY.md`](SECURITY.md).

## License

Project code: MIT. Adapted packing portions retain their BSD-2-Clause notice.
See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
