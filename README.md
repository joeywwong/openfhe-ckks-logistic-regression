# OpenFHE CKKS Logistic Regression Training: Packing and Bootstrapping

> Tested with OpenFHE 1.1.2

OpenFHE 1.1.2 is intentionally pinned to preserve comparability with the
original lab environment. Migrating and revalidating the experiment on a
current OpenFHE release is future work.

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
is optional; ordinary gradient descent (GD) remains the default. NAG users can
retain the existing separate state or put the complete theta/phi state in one
ciphertext, as in the example.

## Improvements over the original lab

| Aspect | Original lab implementation | Current OpenFHE project |
|---|---|---|
| Implementation | Python/TenSEAL | C++17/OpenFHE |
| Ciphertext bootstrapping/refresh | ❌: 'Simulated bootstrapping' - decrypt and re-encrypt with the secret key | ✅: Genuine non-interactive `EvalBootstrap` |
| SIMD packing for training-data | ❌: One feature ciphertext and one label ciphertext per sample | ✅: Row-major SIMD blocks; Framingham input reduced from 1,560 to 14 ciphertexts |
| Optimizer | Full-batch gradient descent | Full-batch GD (default) or fixed-momentum Nesterov accelerated gradient |
| Packing of model parameters, NAG state | ❌: One ciphertext per model parameter, NAG state | Four separate theta/phi weight/bias ciphertexts or one packed ciphertext |
| Sigmoid approximation | Cubic approximation | Original cubic (default) or degree-59 Chebyshev approximation (more accurate, but additional multiplication depth) |

## Project highlights

- **End-to-end encrypted training:** the project evaluates full-batch
  logistic-regression updates on CKKS ciphertexts and continues training after
  genuine non-interactive `EvalBootstrap` refreshes.
- **Genuine bootstrapping and an explicit confidentiality boundary:** compare
  OpenFHE's non-interactive `EvalBootstrap`, which refreshes the encrypted model
  state without decrypting it, with the lab's decrypt-and-re-encrypt workaround.
  The workaround is faster, but it requires secret-key
  access and exposes the plaintext model parameters to the secret-key holder.
- **SIMD sample packing and substantially lower training time:** row-major
  dataset packing, multiple ciphertext blocks, and padding masks reduce the
  encrypted Framingham training input from 1,560 ciphertexts to 14, while
  ensuring that all 780 training rows contribute exactly once. Encrypted
  training time decreased from approximately 230 seconds per epoch without
  sample packing to approximately 8.5 seconds per epoch with sample packing,
  which is about 27x faster. With NAG state packing, the preliminary runtime
  was approximately 4.5 seconds per epoch, about 51x faster than the historical
  unpacked measurement.
- **Improvements over the original lab:** compare the lab's original training
  configuration with current alternatives and measure their effects on runtime,
  convergence, ciphertext count, bootstrapping, consumed levels, numerical
  accuracy, and model quality:
  1. gradient descent (GD) used in the original lab vs fixed-momentum
     Nesterov accelerated gradient (NAG);
  2. model parameters and NAG states separated into two/four ciphertexts vs
     the complete NAG state packed into one ciphertext;
  3. the cubic sigmoid approximation used in the lab vs a degree-59 Chebyshev
     approximation.
- **Plaintext-referenced validation:** integration tests compare every
  encrypted epoch with an independent plaintext optimizer, exercise both
  refresh methods and sigmoid circuits, and continue after a real bootstrap.

The newest optimization packs the complete periodic NAG state—theta, phi,
weights, and intercept—into one ciphertext instead of four. In preliminary
single-run, 20-epoch measurements, it produced the following result:

| Dataset | Separate NAG state | Packed NAG state | Observed total-time reduction | Observed speedup |
|---|---:|---:|---:|---:|
| Framingham | 231.281 s | 85.462 s | 63.0% | 2.71x |
| LogReg sample | 137.176 s | 45.456 s | 66.9% | 3.02x |

Both layouts reached the same final test accuracy; their final training losses
differed by less than `9e-8`. These are local observations, not statistically
established or production-secure performance claims. The runs were not
repeated or order-balanced. See the
[NAG state-packing report](docs/NAG_STATE_PACKING_RESULTS.md) for the timing
breakdown, accuracy checks, trade-offs, raw-data links, and limitations.

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

## Logistic regression

This project trains binary logistic-regression models while the training data
and model state are represented with CKKS ciphertexts.

For a sample $x_i \in \mathbb{R}^d$, binary label $y_i \in \{0,1\}$, weight
vector $w \in \mathbb{R}^d$, and bias $b$, the model first computes the linear
score

```math
z_i = w^\top x_i + b.
```

Standard logistic regression maps this score to a probability using the
logistic sigmoid:

```math
\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
```

The implementation predicts class 1 when the linear score is non-negative:

```math
\widehat{y}_i =
\begin{cases}
1, & z_i \ge 0, \\
0, & z_i < 0.
\end{cases}
```

Because $\sigma(0)=0.5$, this is equivalent to thresholding the exact logistic
probability at 0.5.

### Loss

Reported training loss uses the exact sigmoid and mean binary cross-entropy:

```math
L(w,b)
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log\sigma(z_i)
+
(1-y_i)\log\left(1-\sigma(z_i)\right)
\right].
```

The implementation clamps probabilities slightly away from exactly 0 and 1
before taking logarithms for numerical stability.

This exact sigmoid is used for **reported loss only**. The homomorphic training
circuit instead uses one of the polynomial approximations described below.

### Gradient used for training

With the exact sigmoid, the full-batch logistic-regression gradients are

```math
\nabla_w L
=
\frac{1}{n}
\sum_{i=1}^{n}
x_i\left(\sigma(z_i)-y_i\right),
```

and

```math
\frac{\partial L}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(\sigma(z_i)-y_i\right).
```

Direct evaluation of the exponential inside the sigmoid is not suitable for the
CKKS arithmetic circuit used here. The trainer therefore replaces
$\sigma$ with a polynomial approximation $\widetilde{\sigma}$.

The actual update direction evaluated by both the plaintext reference and the
encrypted trainer is consequently

```math
\widetilde{g}_w
=
\frac{1}{n}
\sum_{i=1}^{n}
x_i\left(\widetilde{\sigma}(z_i)-y_i\right),
```

```math
\widetilde{g}_b
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(\widetilde{\sigma}(z_i)-y_i\right).
```

For the complete training matrix $X$ and label vector $y$, the same computation
can be written as

```math
z=Xw+b\mathbf{1},
\qquad
e=\widetilde{\sigma}(z)-y,
```

```math
\widetilde{g}_w=\frac{1}{n}X^\top e,
\qquad
\widetilde{g}_b=\frac{1}{n}\mathbf{1}^\top e.
```

The encrypted implementation accumulates contributions from every packed
ciphertext block before performing one update and divides by the actual number
of training samples rather than the number of padded CKKS rows. It therefore
remains a full-batch optimizer despite splitting the dataset across multiple
ciphertexts.

### Gradient descent

Ordinary gradient descent updates the model once per complete training batch:

```math
w_{t+1}
=
w_t-\eta\,\widetilde{g}_{w,t},
\qquad
b_{t+1}
=
b_t-\eta\,\widetilde{g}_{b,t},
```

where $\eta$ is the learning rate.

GD is the default optimizer so that the original lab configuration remains
directly reproducible.

### Nesterov accelerated gradient

Nesterov accelerated gradient (NAG) is available as an alternative optimizer.
It evaluates the gradient at a look-ahead model and uses the difference between
successive unaccelerated gradient steps to introduce momentum.

This project follows the
[OpenFHE logistic-regression example](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/lr_nag.cpp#L436-L478)
and uses fixed momentum after the first epoch.

For $t=0,1,\ldots$,

```math
\begin{aligned}
\theta_0 &= \phi_0 = \theta_{\mathrm{init}},
\qquad
\theta_{\mathrm{init}} = 0
\text{ in this implementation}, \\
\beta_t &=
\begin{cases}
0, & t=0, \\
\mu, & t>0,
\end{cases} \\
\phi_{t+1}
&=
\theta_t-\eta\,g(\theta_t), \\
\theta_{t+1}
&=
\phi_{t+1}
+
\beta_t\left(\phi_{t+1}-\phi_t\right).
\end{aligned}
```

where:

- $\theta_t$ is the look-ahead model used to compute the gradient;
- $\phi_t$ is the previous unaccelerated gradient-step model;
- $\eta$ is the learning rate;
- $\mu$ is the configured momentum coefficient;
- $g(\theta_t)$ is the full-batch update direction using the selected sigmoid
  approximation.

The first epoch is an ordinary gradient step because $\beta_0=0$.

Select NAG with `--optimizer nag`. The default momentum is `0.1`, it must be
finite and in `[0,1)`, and a momentum of zero reduces the recurrence to GD.

Compared with GD, nonzero-momentum NAG may reach a target loss in fewer epochs,
but this is not guaranteed for the fixed-momentum implementation at the same
learning rate. In encrypted training it also retains two optimizer states
instead of one and introduces additional homomorphic arithmetic.

## Sigmoid approximation

The training sigmoid is selected with `--sigmoid chebyshev|cubic`.

The default is the cubic polynomial used in the original lab:

```math
\widetilde{\sigma}_{\mathrm{cubic}}(z)
=
0.5+0.197z-0.004z^3.
```

The alternative is the degree-59 Chebyshev approximation of the logistic
function over `[-16,16]` used by the official OpenFHE logistic-regression
example.

| Choice | Training sigmoid | Post-bootstrap levels, separate / packed NAG | Total depth, separate / packed NAG |
|---|---|---:|---:|
| `cubic` (default) | Original lab polynomial `0.5 + 0.197*x - 0.004*x^3` | 10 / 12 | 29 / 31 |
| `chebyshev` | Degree-59 Chebyshev series on `[-16, 16]` | 16 / 18 | 35 / 37 |

The selection applies to both plaintext and encrypted training, with either
GD or NAG and either refresh method. The cubic option is the original
polynomial and is not a degree-3 Chebyshev fit. Neither approximation is
clamped.

The Chebyshev approximation provides greater approximation accuracy over its
target interval but requires substantially greater multiplicative depth than
the cubic circuit.

Reported loss always uses the exact sigmoid, and accuracy always classifies at
linear score zero.

```bash
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 --sigmoid chebyshev
./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 --sigmoid cubic

# Both lab datasets:
SIGMOID=cubic EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

## Sample packing

Each ciphertext contains sample rows padded to a power-of-two feature width.
Weights repeat across rows and labels repeat across columns.

`EvalSumCols` computes row-wise scores and `EvalSumRows` aggregates gradients
across samples. Gradient contributions from all ciphertext blocks are added
before one full-batch optimizer update.

Separate mode masks padded rows out of the bias gradient; packed NAG instead
uses a zero intercept in padded rows.

| Dataset | Training rows | Separate row width / blocks / input CTs | Packed NAG row width / blocks / input CTs |
|---|---:|---:|---:|
| LogReg sample | 700 | 2 / 1 / 2 | 4 / 2 / 4 |
| Framingham | 780 | 16 / 7 / 14 | 16 / 7 / 14 |

The existing ring dimension remains 4,096. Data use all 2,048 slots. The
separate layout keeps the 16-slot sparse bootstrap.

Packed NAG uses at least two model rows (32 slots for the 16-wide Framingham
model) and combines bias with weights as an intercept coordinate.

Multiplicative depth is selected according to the sigmoid approximation and NAG
storage layout shown above. See
[`docs/DESIGN.md`](docs/DESIGN.md#packed-ciphertext-layout) and the
[historical cubic packed results](docs/PACKED_RESULTS.md).

## NAG state packing

The encrypted NAG state can be selected with
`--nag-packing separate|packed`.

- `separate` is the default representation. Theta and phi each have an
  encrypted weight vector and encrypted bias, giving four periodic
  ciphertexts.
- `packed` adapts the technique used by the official OpenFHE example and stores
  both complete optimizer states in a single ciphertext.

In packed mode, bias becomes an intercept coordinate, so each complete model
row is

```text
[weights, bias, padding].
```

One ciphertext alternates complete optimizer-state rows as

```text
[theta][phi][theta][phi]...
```

"Even" and "odd" refer to row blocks rather than individual CKKS slots.

A public theta mask contains ones over theta rows and zeros over phi rows. The
complementary phi mask does the reverse. Multiplying by the corresponding mask
isolates one optimizer state, and adding a copy rotated by one row width fills
the missing rows.

Let $S_t$ denote the packed state, $R$ the row width, and
$M_\theta,M_\phi$ the complementary public masks. The row-cloned states are

```math
\widetilde{\theta}_t
=
M_\theta\odot S_t
+
\mathrm{Rot}_{+R}
\left(M_\theta\odot S_t\right),
```

```math
\widetilde{\phi}_t
=
M_\phi\odot S_t
+
\mathrm{Rot}_{-R}
\left(M_\phi\odot S_t\right),
```

where $\odot$ denotes slot-wise multiplication.

After applying the NAG recurrence, the updated states are combined again as

```math
S_{t+1}
=
M_\theta\odot\widetilde{\theta}_{t+1}
+
M_\phi\odot\widetilde{\phi}_{t+1}.
```

The packed representation therefore retains and bootstraps the complete theta
and phi states in one ciphertext rather than four.

Packed mode requires two model rows in the sparse bootstrap payload and
reserves two additional post-bootstrap levels for extracting and repacking the
optimizer states.

### Encrypted-computation trade-offs

Compared with separate NAG state:

- **State size:** packed mode reduces four periodic NAG ciphertexts to one.
- **Bootstrapping:** separate mode can require four `EvalBootstrap` calls,
  whereas packed mode refreshes one ciphertext.
- **Arithmetic:** packed mode requires masks, rotations, state extraction, and
  repacking.
- **Levels:** these additional operations reserve two more post-bootstrap
  levels.
- **Bias representation:** packed mode treats bias as an intercept coordinate
  and updates it together with the weights.

See
[`docs/NAG_STATE_PACKING_RESULTS.md`](docs/NAG_STATE_PACKING_RESULTS.md)
for the preliminary performance measurements.

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
sigmoid approximations.

Packed NAG is exercised across the 129-row Framingham block boundary as well as
on the LogReg layout.

The tests use a logistic-loss sensitivity bound derived from the measured
coefficient error, rather than an arbitrary fixed tolerance for randomized CKKS
bootstrapping. They also reject mismatched plaintext references.

No CMake presets are needed. The workflow was verified with CMake 3.22.1;
the CMake 3.5.1 compatibility branch has not been executed locally.

### Docker

Docker provides the pinned OpenFHE 1.1.2 dependency and the project build in a
multi-stage image. The OpenFHE source is pinned to the exact commit referenced
by its `v1.1.2` tag.

Build the runtime image with:

```bash
docker build --tag openfhe-logreg:local .
docker run --rm openfhe-logreg:local
```

The default command prints the CLI help. Pass the normal program options after
the image name. For example, this short run exercises encrypted training with
simulated refresh:

```bash
docker run --rm openfhe-logreg:local \
  --dataset logreg --refresh simulated --epochs 1 \
  --output /tmp/docker-smoke.csv
```

To keep the result CSV, mount the host `results` directory at
`/opt/openfhe-lab/results` and select an output path below that directory.

For example, from Bash or WSL:

```bash
mkdir -p results
docker run --rm \
  --mount "type=bind,source=$(pwd)/results,target=/opt/openfhe-lab/results" \
  openfhe-logreg:local \
  --dataset logreg --refresh simulated --epochs 1 \
  --output /opt/openfhe-lab/results/docker-smoke.csv
```

The dedicated `test` target builds the project and runs the complete CTest
suite, including encrypted integration and real-bootstrapping tests:

```bash
docker build --target test --tag openfhe-logreg:test .
```

OpenFHE and project compilation default to two parallel jobs to avoid excessive
memory use. Override that when the Docker host has sufficient resources:

```bash
docker build --build-arg BUILD_JOBS=4 --tag openfhe-logreg:local .
```

## Run the comparison

### Controlled gradient descent versus Nesterov accelerated gradient comparison

Use the paired runner to compare convergence and runtime under identical data
splits, initialization, learning rate, epoch count, datasets, and refresh
methods.

It preserves every raw per-epoch CSV and alternates whether GD or NAG runs
first, reducing systematic warm-cache and first-run bias. With the default four
repeats, each optimizer runs first twice, which gives a balanced result.

For a short verification experiment:

```bash
REPEATS=4 EPOCHS=4 DATASET=all REFRESH=both \
  ./scripts/run_gd_nag_comparison_wsl.sh
```

The defaults are `REPEATS=4`, `EPOCHS=100`, `MOMENTUM=0.1`,
`LEARNING_RATE=0.01`, `DATASET=all`, and `REFRESH=both`.

A full run includes real CKKS bootstrapping and can take a long time.
`RESULT_DIR` selects an output directory; otherwise a timestamped directory is
created under `results/`.

Set `BUILD_AND_TEST=0` to reuse an existing successful build.

Each result directory contains:

- `raw/run_NNN_gd.csv` and `raw/run_NNN_nag.csv`: original per-epoch results;
- `per_run_metrics.csv`: final/minimum loss, final accuracy, timing breakdown,
  refresh count, CKKS levels, and encrypted/plaintext model error;
- `per_run_comparison.csv`: fixed-epoch differences and the first NAG epoch/time
  that reaches the matching GD run's final loss;
- `aggregate_epoch_metrics.csv`: mean and sample standard deviation per epoch
  for plotting loss/accuracy against epochs or cumulative training time;
- `aggregate_optimizer_metrics.csv`: mean and sample standard deviation for
  every optimizer metric;
- `aggregate_comparison.csv`: mean GD/NAG differences, fixed-epoch runtime
  ratio, target-loss success rate, epoch savings, and target-loss speedup.

`experiment_config.csv` records the controlled inputs.

Reported total time is the sum of encrypted arithmetic and optimizer-state
refresh time; common context setup and data encryption are intentionally
excluded. Metric decryption and the discarded paired-refresh measurement remain
separate columns.

In the comparison files, positive `nag_final_loss_improvement` means NAG has
lower loss. Runtime ratios and speedups are `GD / NAG`, so values greater than
one favor NAG.

Test accuracy should be interpreted alongside loss because its discrete
threshold can remain unchanged while optimization improves.

Existing raw result pairs can be summarized again without rerunning OpenFHE:

```bash
python3 scripts/summarize_gd_nag.py \
  --input-dir results/gd_nag_EXPERIMENT/raw \
  --output-dir results/gd_nag_EXPERIMENT
```

See [`docs/GD_NAG_COMPARISON.md`](docs/GD_NAG_COMPARISON.md) for the controlled
methodology and the checked-in two-repeat, four-epoch smoke measurement.

See [`docs/NAG_STATE_PACKING_RESULTS.md`](docs/NAG_STATE_PACKING_RESULTS.md) for
the preliminary separate-versus-packed NAG state measurements.

The experiment retains the lab default of 100 epochs for both datasets and both
refresh methods. Packing reduces the number of encrypted operations.

For a four-epoch verification run covering both refresh methods:

```bash
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

In the integration tests, the degree-59 circuit first bootstraps in epoch 2;
the cubic circuit first bootstraps in epoch 3. Both then bootstrap after each
subsequent epoch.

From a fresh encryption, GD reaches consumed level 10 with Chebyshev or 6 with
cubic. Further real-mode epochs consume 11 or 7 levels, respectively; nonzero
NAG momentum adds one level after the first epoch.

Refresh is triggered by actual consumed levels, not a fixed epoch number.

Tests run through epoch 3 for Chebyshev and epoch 4 for cubic to verify training
after the first real bootstrap.

To run the lab's full 100 epochs:

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

./build/openfhe_lab_compare --dataset logreg --refresh both --epochs 4 \
  --optimizer nag --momentum 0.1 --nag-packing packed

# Both lab datasets:
OPTIMIZER=nag MOMENTUM=0.1 NAG_PACKING=packed EPOCHS=4 \
  ./scripts/run_comparison_wsl.sh
```

Options:

```text
--dataset logreg|framingham|all
--refresh simulated|real|both
--epochs N
--learning-rate X
--optimizer gd|nag
--momentum X
--nag-packing separate|packed
--sigmoid chebyshev|cubic
--output PATH
```

New measurements go to `results/benchmark_packed_<sigmoid>.csv` for GD or
`results/benchmark_nag_<sigmoid>.csv` for separate NAG.

Packed NAG uses `results/benchmark_nag_packed_<sigmoid>.csv`, where
`<sigmoid>` is `chebyshev` or `cubic`.

Use `--output` with the executable or `OUTPUT_PATH` with the script to override
the path. The script also accepts `SIGMOID` (default: `cubic`) and
`NAG_PACKING` (default: `separate`).

CSV rows include `optimizer`, the effective `momentum` (zero for GD), `sigmoid`,
and `nag_packing`.

Existing result files and reports are historical measurements. See
[`docs/PACKED_RESULTS.md`](docs/PACKED_RESULTS.md) for the earlier packed
cubic-sigmoid GD run; [`docs/RESULTS.md`](docs/RESULTS.md) is the historical
unpacked report.

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

NAG arithmetic and refresh timings include both optimizer states, including
mask/rotation extraction and repacking in packed mode.

The paired simulated-refresh measurement also refreshes a discarded copy of the
complete selected state representation.

## Repository layout

```text
.
├── app/main.cpp
├── data/
│   ├── LogReg_sample_dataset.csv
│   └── framingham.csv
├── docs/
│   ├── DESIGN.md
│   ├── GD_NAG_COMPARISON.md
│   ├── NAG_STATE_PACKING_RESULTS.md
│   ├── PACKED_RESULTS.md
│   └── RESULTS.md
├── include/openfhe_lab/
├── results/
│   ├── benchmark*.csv
│   ├── nag_packed.csv
│   └── nag_separate.csv
├── scripts/
├── src/
└── tests/
```

## Security

The default ring dimension is a laptop demonstration using `HEStd_NotSet`; it
makes no production security claim.

Simulated bootstrapping explicitly exposes the model to the secret-key holder
between epochs.

See [`SECURITY.md`](SECURITY.md).

## License

Project code: MIT.

Adapted packing and NAG portions retain their BSD-2-Clause notice.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).