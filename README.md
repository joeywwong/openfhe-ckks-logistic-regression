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
The forward pass supports that example's degree-59 Chebyshev approximation
of the logistic function over `[-16, 16]`, or the original lab/main-branch
cubic `0.5 + 0.197*x - 0.004*x^3`. The lab cubic is the default.
Full-batch Nesterov accelerated gradient (NAG), adapted from the same example,
is optional; ordinary gradient descent (GD) remains the default.

## Default experiment behavior

- 70% training and 30% test data after a seed-4 shuffle;
- zero-initialized binary logistic regression;
- full-batch gradient descent, 100 epochs, learning rate 0.01;
- original lab cubic sigmoid `0.5 + 0.197*x - 0.004*x^3` in plaintext and ciphertext training;
- encrypted features and labels, now batched into row-major ciphertext blocks;
- separate encrypted weight and bias ciphertexts;
- per-epoch test accuracy and exact-sigmoid training loss after model decryption;
- the lab's exact Framingham column removal, class balancing, and full-dataset
  standardization order.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the one-to-one mapping and timing
definitions.

## Sigmoid approximation

Select the training approximation with `--sigmoid chebyshev|cubic`:

| Choice | Training sigmoid | Levels reserved after bootstrap | Total multiplicative depth |
|---|---|---:|---:|
| `cubic` (default) | Original lab/main polynomial `0.5 + 0.197*x - 0.004*x^3` | 10 | 29 |
| `chebyshev` | Degree-59 Chebyshev series on `[-16, 16]` | 16 | 35 |

The selection applies to both plaintext and encrypted training, with either
GD or NAG and either refresh method. The cubic option restores the original
polynomial and encrypted evaluation circuit; it is not a degree-3 Chebyshev
fit. Neither approximation is clamped. Reported loss always uses the exact
sigmoid, and accuracy always classifies at linear score zero.

```bash
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 --sigmoid chebyshev
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 --sigmoid cubic
# Both lab datasets, using the comparison script:
SIGMOID=cubic EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

## Nesterov accelerated gradient

NAG evaluates the gradient at a **look-ahead model**, extrapolated using recent
parameter changes, so the gradient can correct the momentum direction. 

This project follows [the OpenFHE update](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/lr_nag.cpp#L436-L478),
using fixed momentum after the first epoch. For `t = 0, 1, ...`, both weights
and bias follow:

```math
\begin{aligned}
\theta_0 &= \phi_0 = \theta_{\mathrm{init}}, \qquad \theta_{\mathrm{init}} = 0 \text{ in this implementation.}, \\
\beta_t &=
\begin{cases}
0, & t = 0, \\
\mu, & t > 0,
\end{cases} \\
\phi_{t+1} &= \theta_t - \eta\,g(\theta_t), \\
\theta_{t+1} &= \phi_{t+1} + \beta_t\left(\phi_{t+1}-\phi_t\right).
\end{aligned}
```

- $\theta_t$: look-ahead model used to compute the gradient.
- $\phi_t$: previous unaccelerated gradient-step model.
- $\eta$: learning rate.
- $\mu$: configured momentum coefficient.
- $g(\theta_t)$: full-batch gradient using the selected sigmoid approximation.

The first epoch omits extrapolation because $\beta_0=0$.

Select `--optimizer nag`; `--momentum` defaults to `0.1`, must be finite in
`[0, 1)`, and `0` reduces to GD. The sigmoid approximation, full-batch
averaging, and learning rate are identical between optimizers. In a 100-epoch
experiment, NAG showed faster
loss reduction than gradient descent under the tested configuration. 
Faster convergence is not guaranteed for this fixed-momentum, 
approximate-gradient implementation.

With nonzero momentum, encrypted training retains four ciphertexts (weights
and bias for both model states $\theta_t$ and $\phi_t$), preserving both states through simulated or real
bootstrapping. The extra arithmetic and refresh work can outweigh any reduction
in epochs. Larger momentum can also push scores outside the Chebyshev
approximation interval `[-16, 16]`, where approximation guarantees no longer
apply.

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
refreshed separately. Multiplicative depth is selected with the sigmoid:
35 for Chebyshev, or 29 for the lab cubic. See [the design](docs/DESIGN.md#packed-ciphertext-layout) and the
[historical cubic packed results](docs/PACKED_RESULTS.md).

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
The tests also check NAG against an independent velocity formulation, zero
momentum against GD, and encrypted NAG after real bootstrapping, for both
sigmoid approximations. They also reject mismatched plaintext references.
No CMake presets are needed. The workflow was verified with CMake 3.22.1;
the CMake 3.5.1 compatibility branch has not been executed locally.

## Run the comparison

The experiment retains the lab default of 100 epochs for both datasets and both
refresh methods. Packing reduces the number of encrypted operations. For a
four-epoch verification run covering both refresh methods:

```bash
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

In the integration tests, the degree-59 circuit first bootstraps in epoch 2;
the cubic circuit first bootstraps in epoch 3. Both then bootstrap after each
subsequent epoch. From a fresh encryption, GD reaches consumed level 10 with
Chebyshev or 6 with cubic. Further real-mode epochs consume 11 or 7 levels,
respectively; nonzero NAG momentum adds one level after the first epoch.
Refresh is triggered by actual consumed levels, not a fixed epoch number.
Tests run through epoch 3 for Chebyshev and epoch 4 for cubic to verify training
after the first real bootstrap.

To run the lab's full 100 epochs (not represented as measured by the four-epoch report):

```bash
./scripts/run_comparison_wsl.sh
```

Direct executable examples:

```bash
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4
./build/openfhe_lab_compare --dataset framingham --refresh both --epochs 4
```

For NAG with both refresh methods:

```bash
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 --optimizer nag --momentum 0.1
# Both lab datasets, using the comparison script:
OPTIMIZER=nag MOMENTUM=0.1 EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

Options:

```text
--dataset logreg|framingham|all
--refresh simulated|real|both
--epochs N
--learning-rate X
--optimizer gd|nag
--momentum X
--sigmoid chebyshev|cubic
--output PATH
```

New measurements go to `results/benchmark_packed_<sigmoid>.csv` for GD or
`results/benchmark_nag_<sigmoid>.csv` for NAG, where `<sigmoid>` is `chebyshev`
or `cubic`. This keeps the two approximations in separate files by default.
Use `--output` with the executable or `OUTPUT_PATH` with the script to override
the path. The script also accepts `SIGMOID` (default: `cubic`). CSV rows
include `optimizer`, the effective `momentum` (zero for GD), and `sigmoid`.
Existing result files and reports are historical measurements. See
[`docs/PACKED_RESULTS.md`](docs/PACKED_RESULTS.md) for the earlier packed
cubic-sigmoid GD run;
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
- maximum consumed CKKS level across the complete optimizer state before and
  after refresh.

NAG arithmetic and refresh timings include both optimizer states. The paired
simulated-refresh measurement also refreshes a discarded copy of both states.

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

Project code: MIT. Adapted packing and NAG portions retain their BSD-2-Clause notice.
See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
