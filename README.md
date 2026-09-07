# OpenFHE CKKS Logistic Regression Training: Packing and Bootstrapping

> Tested with OpenFHE 1.1.2

This project implements encrypted logistic-regression training in C++ with OpenFHE and CKKS.

It ports my original [`fhe-ckks-lwe-encrypted-ml-lab`](https://github.com/joeywwong/fhe-ckks-lwe-encrypted-ml-lab) from Python/TenSEAL to OpenFHE. It is built from my [`openfhe-template`](https://github.com/joeywwong/openfhe-template).

OpenFHE 1.1.2 is intentionally pinned so that the results remain comparable with the original lab. Migrating and revalidating the project on a newer OpenFHE release is future work.

The project keeps the original lab experiment, then adds:

- genuine non-interactive CKKS bootstrapping with `EvalBootstrap`;
- row-major CKKS SIMD packing for training samples;
- gradient descent (GD) and Nesterov accelerated gradient (NAG);
- separate or one-ciphertext packing of the complete NAG state;
- cubic or degree-59 Chebyshev sigmoid approximation;
- plaintext-reference validation for encrypted training.

The two datasets are `LogReg_sample_dataset.csv` and `framingham.csv`.

## Improvements over the original lab

| Aspect | Original lab | Current OpenFHE project |
|---|---|---|
| Implementation | Python/TenSEAL | C++17/OpenFHE |
| Model refresh | Decrypt and re-encrypt | Simulated refresh or genuine `EvalBootstrap` |
| Training-data packing | One feature ciphertext and one label ciphertext per sample | Row-major SIMD blocks |
| Framingham encrypted input | 1,560 ciphertexts | 14 ciphertexts |
| Optimizer | Full-batch GD | Full-batch GD or NAG |
| NAG state | Not used | Four separate ciphertexts or one packed ciphertext |
| Sigmoid approximation | Cubic polynomial | Cubic or degree-59 Chebyshev |
| Validation | Lab experiment | Encrypted epochs checked against a plaintext reference |

## Project highlights

- **Genuine CKKS bootstrapping.** OpenFHE `EvalBootstrap` refreshes the encrypted model without decrypting it.

- **Explicit comparison with simulated bootstrapping.** The original lab workaround decrypts and re-encrypts the model. This is useful as a timing baseline, but it requires the secret key and exposes the plaintext model to the key holder.

- **SIMD sample packing and substantially lower training time:** row-major
  dataset packing, multiple ciphertext blocks, and padding masks reduce the
  encrypted Framingham training input from 1,560 ciphertexts to 14, while
  ensuring that all 780 training rows contribute exactly once. Encrypted
  training time decreased from approximately 230 seconds per epoch without
  sample packing to approximately 8.5 seconds per epoch with sample packing,
  which is about 27x faster. With NAG state packing, the preliminary runtime
  was approximately 4.5 seconds per epoch, about 51x faster than the historical
  unpacked measurement.

- **Full-batch training across multiple ciphertext blocks.** Gradients from all blocks are accumulated before one model update. Padded rows do not change the training result.

- **Selectable optimizer.** GD remains the default. Fixed-momentum NAG is also available.

- **One-ciphertext NAG state.** The complete theta/phi optimizer state can be packed into one ciphertext instead of four periodically refreshed ciphertexts.

- **Selectable sigmoid circuit.** Training can use the original cubic approximation or the degree-59 Chebyshev approximation used by the official OpenFHE logistic-regression example.

- **Plaintext-referenced tests.** Encrypted epochs are compared with an independent plaintext optimizer. Tests cover packing, padding, both sigmoid circuits, NAG, and training after genuine bootstrapping.

## Performance observations

### Sample packing

For the Framingham experiment, SIMD sample packing reduced the encrypted training input from 1,560 ciphertexts to 14.

The historical unpacked implementation required about 230 seconds per epoch on the local test machine. The packed implementation requires about 8.5 seconds per epoch in the corresponding local measurements. This is about a 27x historical speedup.

The measurements are specific to this implementation and machine. They are not general OpenFHE performance claims.

### NAG state packing

The newest optimization packs the complete NAG theta/phi state into one ciphertext.

Preliminary 20-epoch measurements with real CKKS bootstrapping gave:

| Dataset | Separate NAG state | Packed NAG state | Time reduction | Speedup |
|---|---:|---:|---:|---:|
| Framingham | 231.281 s | 85.462 s | 63.0% | 2.71x |
| LogReg sample | 137.176 s | 45.456 s | 66.9% | 3.02x |

Both layouts reached the same displayed final test accuracy. Their final training losses differed by less than `9e-8`.

These are preliminary single-run observations. The runs were not repeated or order-balanced. See [`docs/NAG_STATE_PACKING_RESULTS.md`](docs/NAG_STATE_PACKING_RESULTS.md) for the timing breakdown, numerical checks, trade-offs, and limitations.

## Default experiment

The defaults preserve the original lab as closely as possible:

- 70% training and 30% test data after a seed-4 shuffle;
- zero-initialized binary logistic regression;
- full-batch gradient descent;
- 100 epochs;
- learning rate `0.01`;
- cubic training sigmoid `0.5 + 0.197*x - 0.004*x^3`;
- encrypted features and labels;
- separate encrypted weight and bias ciphertexts;
- exact-sigmoid training loss for reporting;
- test accuracy after model decryption;
- the original Framingham preprocessing and class-balancing order.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full mapping to the original lab.

## Logistic regression

The project trains a binary logistic-regression model while the training data and model state are represented with CKKS ciphertexts.

For a sample $x_i \in \mathbb{R}^d$, label $y_i \in \{0,1\}$, weights $w$, and bias $b$, the linear score is

```math
z_i = w^\top x_i + b.
```

The exact logistic sigmoid is

```math
\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
```

The implementation predicts class 1 when

```math
z_i \ge 0.
```

Since $\sigma(0)=0.5$, this is equivalent to thresholding the exact sigmoid at 0.5.

### Loss

Reported training loss uses mean binary cross-entropy:

```math
L(w,b)
=
-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log\sigma(z_i)
+
(1-y_i)\log(1-\sigma(z_i))
\right].
```

Probabilities are moved slightly away from exactly 0 and 1 before the logarithm for numerical stability.

The exact sigmoid is used for **reported loss only**. Homomorphic training uses a polynomial approximation.

### Gradient used for training

With the exact sigmoid, the full-batch gradients are

```math
\nabla_w L
=
\frac{1}{n}
\sum_{i=1}^{n}
x_i(\sigma(z_i)-y_i),
```

and

```math
\frac{\partial L}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\sigma(z_i)-y_i).
```

Directly evaluating the exponential is not suitable for the CKKS arithmetic circuit used here.

Training therefore replaces $\sigma$ with a polynomial approximation $\widetilde{\sigma}$.

For the training matrix $X$ and labels $y$,

```math
z=Xw+b\mathbf{1},
\qquad
e=\widetilde{\sigma}(z)-y,
```

and

```math
\widetilde{g}_w
=
\frac{1}{n}X^\top e,
\qquad
\widetilde{g}_b
=
\frac{1}{n}\mathbf{1}^\top e.
```

The encrypted trainer computes the same full-batch update. If the dataset spans several ciphertext blocks, it first adds the gradients from every block.

It divides by the real number of training samples, not by the number of padded CKKS rows.

### Gradient descent

Gradient descent updates the model as

```math
w_{t+1}
=
w_t-\eta\widetilde{g}_{w,t},
\qquad
b_{t+1}
=
b_t-\eta\widetilde{g}_{b,t},
```

where $\eta$ is the learning rate.

GD is the default optimizer because it matches the original lab.

### Nesterov accelerated gradient

Nesterov accelerated gradient is available with

```text
--optimizer nag
```

The implementation follows the fixed-momentum recurrence used by the official [OpenFHE logistic-regression example](https://github.com/openfheorg/openfhe-logreg-training-examples).

For $t=0,1,\ldots$,

```math
\begin{aligned}
\theta_0 &= \phi_0 = 0,\\
\beta_t &=
\begin{cases}
0, & t=0,\\
\mu, & t>0,
\end{cases}\\
\phi_{t+1}
&=
\theta_t-\eta g(\theta_t),\\
\theta_{t+1}
&=
\phi_{t+1}
+
\beta_t(\phi_{t+1}-\phi_t).
\end{aligned}
```

Here:

- $\theta_t$ is the model used to evaluate the gradient;
- $\phi_t$ is the previous unaccelerated gradient-step model;
- $\eta$ is the learning rate;
- $\mu$ is the momentum coefficient.

The first epoch is an ordinary gradient step because $\beta_0=0$.

The default momentum is `0.1`. It must be in `[0,1)`. Momentum `0` reduces the recurrence to GD.

NAG may reduce the number of epochs needed to reach a target loss. This is not guaranteed for every dataset or learning rate. It also requires additional encrypted optimizer state.

### From logistic-regression algebra to packed OpenFHE operations

The packed implementation maps the logistic-regression computation to CKKS SIMD operations as follows:

| Logistic-regression step | Algebra | Packed OpenFHE operation |
|---|---|---|
| Linear feature products | $x_i \odot w$ | `EvalMult` |
| Dot product | $x_i^\top w$ | `EvalSumCols` within each packed row |
| Add bias | $x_i^\top w+b$ | ciphertext addition |
| Approximate sigmoid | $\widetilde{\sigma}(z_i)$ | cubic circuit or `EvalLogistic` |
| Error | $\widetilde{\sigma}(z_i)-y_i$ | ciphertext subtraction |
| Weight-gradient terms | $x_i e_i$ | `EvalMult` |
| Sum over samples | $\sum_i x_i e_i$ | `EvalSumRows` |
| Bias gradient | $\sum_i e_i$ | masked `EvalSumRows`, or intercept coordinate in packed NAG |
| Full-batch update | $\theta-\eta g$ | ciphertext/plaintext multiplication and addition |
| NAG extrapolation | $\phi_{t+1}+\mu(\phi_{t+1}-\phi_t)$ | ciphertext subtraction, scalar multiplication, and addition |

This mapping is described in more detail in [`docs/DESIGN.md`](docs/DESIGN.md#packed-ciphertext-layout).

## Sigmoid approximation

Choose the training sigmoid with

```text
--sigmoid cubic|chebyshev
```

### Cubic

The default is the polynomial used by the original lab:

```math
\widetilde{\sigma}_{\mathrm{cubic}}(z)
=
0.5+0.197z-0.004z^3.
```

This polynomial is not a degree-3 Chebyshev fit.

### Chebyshev

The alternative is the degree-59 Chebyshev approximation of the logistic function over `[-16,16]` used by the official OpenFHE logistic-regression example.

| Choice | Approximation | Post-bootstrap levels, separate / packed NAG | Total depth, separate / packed NAG |
|---|---|---:|---:|
| `cubic` | `0.5 + 0.197*x - 0.004*x^3` | 10 / 12 | 29 / 31 |
| `chebyshev` | Degree 59 on `[-16,16]` | 16 / 18 | 35 / 37 |

The Chebyshev approximation is more accurate over its target interval, but it needs more multiplicative depth.

The selected approximation is used by both the plaintext reference and encrypted trainer.

Reported loss still uses the exact sigmoid.

Example:

```bash
./build/openfhe_lab_compare \
  --dataset logreg --refresh both --epochs 4 \
  --sigmoid chebyshev
```

## Sample packing

Training samples use row-major CKKS SIMD packing adapted from the official OpenFHE logistic-regression example.

Each ciphertext contains several sample rows. Each row is padded to a power-of-two width.

For the separate model layout:

```text
features: [x00,x01 | x10,x11 | x20,x21 | 0,0 | ...]
labels:   [y0, y0  | y1, y1  | y2, y2  | 0,0 | ...]
weights:  [w0, w1  | w0, w1  | w0, w1  | ...]
bias:     [b,  b   | b,  b   | b,  b   | ...]
```

`EvalSumCols` computes each row's dot product.

After the sigmoid and error calculation, `EvalSumRows` adds gradient contributions across sample rows.

If a dataset needs several ciphertext blocks, gradients from all blocks are added before one optimizer update. The algorithm therefore remains full-batch training.

Padded rows are excluded from the bias gradient.

| Dataset | Training rows | Separate row width / blocks / input CTs | Packed NAG row width / blocks / input CTs |
|---|---:|---:|---:|
| LogReg sample | 700 | 2 / 1 / 2 | 4 / 2 / 4 |
| Framingham | 780 | 16 / 7 / 14 | 16 / 7 / 14 |

The demonstration ring dimension is 4,096, giving 2,048 logical CKKS slots.

See [`docs/DESIGN.md`](docs/DESIGN.md#packed-ciphertext-layout) for the exact layout, masks, padding rules, and sparse-bootstrap configuration.

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

## Refresh methods

The project compares two model-refresh methods.

### Simulated bootstrapping

The updated encrypted model is decrypted and encrypted again after an epoch.

This reproduces the workaround used in the original TenSEAL lab.

It restores a fresh ciphertext, but it requires the secret key. The model is visible in plaintext to the secret-key holder during refresh.

### Real bootstrapping

OpenFHE's `EvalBootstrap` refreshes a worn CKKS ciphertext without decrypting it.

The implementation waits until enough natural levels have been consumed for `EvalBootstrap` to produce a genuinely refreshed ciphertext. Refresh is therefore triggered by ciphertext level, not by a hard-coded epoch number.

The real training branch continues only from the bootstrapped ciphertext. Any decryptions used for metrics or paired timing measurements do not feed back into training.

Detailed level and timing rules are documented in [`docs/DESIGN.md`](docs/DESIGN.md#refresh-methods).

## Build and test

### Requirements

- Ubuntu 22.04 or a comparable Linux/WSL environment;
- CMake 3.5.1 or later;
- a C++17 compiler;
- OpenFHE 1.1.2.

From WSL:

```bash
git clone https://github.com/joeywwong/openfhe-ckks-logistic-regression.git
cd openfhe-ckks-logistic-regression
./scripts/build_and_test_wsl.sh
```

The test suite includes:

- plaintext logistic-regression tests;
- sample-packing and padding tests;
- GD and NAG checks;
- zero-momentum NAG versus GD;
- cubic and Chebyshev sigmoid circuits;
- separate and packed NAG state;
- encrypted/plaintext agreement;
- real CKKS bootstrapping;
- continued encrypted training after bootstrap;
- multi-block Framingham packing.

## Docker

Docker provides a reproducible OpenFHE 1.1.2 build.

Build the runtime image:

```bash
docker build --tag openfhe-logreg:local .
```

Show the CLI help:

```bash
docker run --rm openfhe-logreg:local
```

Run a short encrypted experiment:

```bash
docker run --rm openfhe-logreg:local \
  --dataset logreg \
  --refresh simulated \
  --epochs 1 \
  --output /tmp/docker-smoke.csv
```

Build the image and run the full CTest suite:

```bash
docker build --target test --tag openfhe-logreg:test .
```

OpenFHE and project compilation use two parallel jobs by default to limit memory use.

A host with more memory can use:

```bash
docker build \
  --build-arg BUILD_JOBS=4 \
  --tag openfhe-logreg:local .
```

## Run experiments

A short run with both datasets and refresh methods:

```bash
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

The default script runs 100 epochs.

Direct examples:

```bash
./build/openfhe_lab_compare \
  --dataset logreg \
  --refresh both \
  --epochs 4
```

```bash
./build/openfhe_lab_compare \
  --dataset framingham \
  --refresh both \
  --epochs 4
```

Run NAG:

```bash
./build/openfhe_lab_compare \
  --dataset logreg \
  --refresh both \
  --epochs 4 \
  --optimizer nag \
  --momentum 0.1
```

Run NAG with one-ciphertext state packing:

```bash
./build/openfhe_lab_compare \
  --dataset logreg \
  --refresh both \
  --epochs 4 \
  --optimizer nag \
  --momentum 0.1 \
  --nag-packing packed
```

Use the Chebyshev sigmoid:

```bash
./build/openfhe_lab_compare \
  --dataset logreg \
  --refresh both \
  --epochs 4 \
  --sigmoid chebyshev
```

### CLI options

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

## Controlled GD versus NAG comparison

The repository also provides a paired GD/NAG runner.

It keeps the dataset split, initialization, learning rate, epoch count, dataset, and refresh method the same for both optimizers.

It alternates which optimizer runs first. This reduces systematic first-run or warm-cache bias.

For a short experiment:

```bash
REPEATS=4 EPOCHS=4 DATASET=all REFRESH=both \
  ./scripts/run_gd_nag_comparison_wsl.sh
```

The main defaults are:

```text
REPEATS=4
EPOCHS=100
MOMENTUM=0.1
LEARNING_RATE=0.01
DATASET=all
REFRESH=both
```

Each optimizer runs first twice when `REPEATS=4`.

The runner preserves raw per-epoch results and produces per-run and aggregate summaries.

See [`docs/GD_NAG_COMPARISON.md`](docs/GD_NAG_COMPARISON.md) for the methodology and checked-in smoke results.

## Reported metrics

CSV output includes:

- encrypted arithmetic time;
- refresh time;
- total training time per epoch;
- metric-only decryption time;
- test accuracy;
- exact-sigmoid training loss;
- maximum encrypted-model error against the matching plaintext epoch;
- paired simulated-refresh time at genuine bootstrap points;
- CKKS level before and after refresh;
- optimizer and momentum;
- sigmoid approximation;
- NAG packing mode.

Context setup and input encryption are not included in per-epoch training time.

Detailed timing definitions are in [`docs/DESIGN.md`](docs/DESIGN.md#timing-definitions).

## Documentation

More detailed material is kept outside the main README:

- [`docs/DESIGN.md`](docs/DESIGN.md) — lab-to-OpenFHE mapping, packing layout, preprocessing, NAG state, refresh behavior, CKKS parameters, and timing definitions.
- [`docs/GD_NAG_COMPARISON.md`](docs/GD_NAG_COMPARISON.md) — controlled GD/NAG comparison methodology and results.
- [`docs/NAG_STATE_PACKING_RESULTS.md`](docs/NAG_STATE_PACKING_RESULTS.md) — separate-versus-packed NAG state measurements and correctness checks.
- [`docs/PACKED_RESULTS.md`](docs/PACKED_RESULTS.md) — historical packed cubic-GD measurements.
- [`docs/RESULTS.md`](docs/RESULTS.md) — historical unpacked measurements.

## Repository layout

```text
.
├── app/
│   └── main.cpp
├── data/
│   ├── LogReg_sample_dataset.csv
│   └── framingham.csv
├── docs/
│   ├── DESIGN.md
│   ├── GD_NAG_COMPARISON.md
│   ├── NAG_STATE_PACKING_RESULTS.md
│   ├── PACKED_RESULTS.md
│   └── RESULTS.md
├── include/
│   └── openfhe_lab/
├── results/
│   ├── benchmark*.csv
│   ├── nag_packed.csv
│   └── nag_separate.csv
├── scripts/
├── src/
└── tests/
```

## Security

This is an educational and research-oriented experiment. It is not a production cryptographic system or medical prediction tool.

The default ring dimension is 4,096 and uses `HEStd_NotSet`. It is a laptop-scale demonstration configuration and makes **no standard production-security claim**.

Real `EvalBootstrap` refreshes encrypted state without exposing the model to the secret-key holder.

Simulated bootstrapping decrypts the model and therefore crosses that confidentiality boundary.

See [`SECURITY.md`](SECURITY.md).

## License

Project code is released under the MIT License.

Adapted packing and NAG portions retain their BSD-2-Clause notice.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).