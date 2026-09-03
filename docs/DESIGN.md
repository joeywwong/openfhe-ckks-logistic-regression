# Faithful lab-to-OpenFHE design

## Scope

This project ports the original TenSEAL experiment to OpenFHE. By default it
retains the lab's optimizer, split, preprocessing, and metrics. Training can
use the original lab/main-branch cubic sigmoid (the default) or the official
OpenFHE example's degree-59 Chebyshev approximation. Nesterov accelerated gradient is
available as an optional optimizer.
At the user's request, the original ciphertext-per-sample layout is now replaced
by sample packing adapted from the official OpenFHE example.
Both refresh methods use the same packed circuit.

## Behavior retained from the lab

| Part | Retained behavior |
|---|---|
| Datasets | Only `LogReg_sample_dataset.csv` and `framingham.csv` |
| Split | Shuffle, then first `floor(0.30*n)` records for test; seed 4 |
| Model | Binary logistic regression; zero-initialized weights and bias |
| Optimizer | Full-batch gradient descent by default; optional NAG |
| Default run | 100 epochs and learning rate 0.01 |
| Encrypted input | Features and labels remain encrypted; multiple samples now share each block |
| Model layout | Encrypted weights and encrypted bias are separate ciphertexts |
| Forward sigmoid | `--sigmoid cubic` (default): lab polynomial `0.5 + 0.197*x - 0.004*x^3`; `--sigmoid chebyshev`: degree 59 on `[-16, 16]` |
| Gradient | `(prediction-label)*x` and `(prediction-label)` for bias |
| Accuracy | Decrypt model, classify the plaintext test set at linear score zero |
| Loss | Exact logistic sigmoid and binary cross-entropy on plaintext training data |

The original lab standardizes the complete balanced Framingham dataset before
splitting it. Although fitting preprocessing before a split is normally changed
to avoid leakage, it is deliberately retained here because this is a
reproduction rather than a redesigned ML evaluation.

## Packed ciphertext layout

The layout is adapted from OpenFHE's
[`utils.cpp`](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/utils.cpp)
(`Mat2CtMRM`, `OneDMat2CtVCC`, and row cloning) and
[`enc_matrix.h`](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/enc_matrix.h)
(`MatrixVectorProductRow` and `MatrixVectorProductCol`).

For the default separate model, width `R` is the next power of two at least
`d`, where `d` is the feature count. A 2,048-slot ciphertext holds `2048/R`
sample rows. For two features and three samples:

```text
features: [x00,x01 | x10,x11 | x20,x21 | 0,0 | ...]
labels:   [y0, y0  | y1, y1  | y2, y2  | 0,0 | ...]
weights:  [w0, w1  | w0, w1  | w0, w1  | w0,w1 | ...]
bias:     [b,  b   | b,  b   | b,  b   | b,b | ...]
valid:    [1,  1   | 1,  1   | 1,  1   | 0,0 | ...]
```

1. Multiply features by repeated weights. `EvalSumCols` sums feature columns
   independently within each row and replicates each dot product over its row.
2. Add the encrypted bias, apply the selected sigmoid, and subtract repeated
   encrypted labels. Chebyshev uses `EvalLogistic(score, -16, 16, 59)`; cubic
   uses the original `0.5 + 0.197*score - 0.004*score^3` multiplication circuit.
3. Multiply errors by feature values. `EvalSumRows` sums sample rows for each
   feature column, producing a repeated weight-gradient row.
4. Mask errors by public row occupancy and use `EvalSumRows` for bias gradients.
   Zero feature padding alone is insufficient: padded rows would otherwise
   contribute `sigmoid(bias)` to the bias gradient.
5. Sum gradients from **all blocks**, divide by the actual training count
   (not padded rows), and make one update with learning rate 0.01.

Packed NAG instead chooses `R` as the next power of two at least `d + 1` and
appends a public intercept coordinate:

```text
features: [x00,x01,1,0 | x10,x11,1,0 | x20,x21,1,0 | 0,0,0,0 | ...]
model:    [w0, w1, b,0 | w0, w1, b,0 | w0, w1, b,0 | w0,w1,b,0 | ...]
```

The same matrix reductions then compute weight and bias gradients together.
Padded rows have a zero intercept, so they make no bias-gradient contribution.

This extends the upstream layout to multiple blocks without changing to
mini-batch SGD. Optional Nesterov acceleration uses the same full-batch
gradient. The upstream Chebyshev sigmoid is selectable alongside the lab
cubic; pre-scaled data and the upstream dataset are not adopted. The upstream code is explicitly illustrative, not a
performance benchmark; measurements here refer only to this local adaptation.

With separate storage, LogReg uses width 2, 1,024 rows/block, one block and two
input ciphertexts. Packed NAG's intercept changes LogReg to width 4, 512
rows/block, two blocks and four input ciphertexts. Framingham uses width 16,
128 rows/block, seven blocks and 14 input ciphertexts in either mode. GD
retains two separate encrypted model ciphertexts. Separate NAG with nonzero
momentum retains four; packed NAG retains one, as described below.

The model repeats with period `R`. Following the upstream sparse-bootstrap
pattern, a clone's slot metadata is reduced for `EvalBootstrap` and restored
to 2,048 for matrix operations. Separate storage uses 16 sparse slots. Packed
storage needs at least two rows, so the 16-wide model uses 32. This is valid
only for the periodic model, not for arbitrary sample ciphertexts.

Packing changes floating-point summation order, not the full-batch formula.
Integration tests compare every encrypted epoch with the matching plaintext
optimizer and sigmoid, covering partial blocks, both feature widths, multiple
blocks, and training after the first real bootstrap.

## Sigmoid selection

`labml::SigmoidApproximation` selects `Chebyshev` or `Cubic`. Pass it as the
last argument to `TrainPlaintext` (after the optimizer) and set
`CkksConfiguration::sigmoid` before `CreateFheRuntime`. The runtime binds the
encrypted circuit to the corresponding depth. `TrainEncrypted` rejects a
plaintext reference that records a different approximation. API calls
without a selection use the lab cubic.

The plaintext Chebyshev path uses the same OpenFHE coefficients as
`EvalLogistic`, evaluated with Clenshaw recurrence. The cubic path uses the
exact lab coefficients in both trainers, without clipping or a new fit.
Only the training gradient uses this approximation; loss and accuracy retain
the lab's evaluation rules.

## Nesterov optimizer state

The optional `--optimizer nag` follows the fixed-momentum theta/phi recurrence
in upstream `lr_nag.cpp`, including its first epoch without momentum. The
plaintext and encrypted trainers both compute the gradient at theta, take an
unaccelerated step phi_next, then extrapolate theta_next from phi_next and the
previous phi. Both weights and bias participate; the reported/final model is
theta. See [the README](../README.md#nesterov-accelerated-gradient) for the formula.

The default `--nag-packing separate` representation retains two models,
each with a weight and bias ciphertext (four ciphertexts total). Each worn
ciphertext is bootstrapped only when its consumed level exceeds the trigger; a
less-worn state can be retained without a no-op bootstrap.

`--nag-packing packed` adapts the upstream
`collateOneDMats2CtVRC` technique. Theta occupies every even row block and phi
every odd row block. Before an epoch, an alternating plaintext mask isolates
one state and a rotation by one signed row width fills its missing blocks:

```text
packed:       [theta][phi][theta][phi]...
theta mask:   [theta][  0][theta][  0]... + rotate(+rowWidth)
phi mask:     [  0][phi][  0][phi]...     + rotate(-rowWidth)
```

After the update, complementary masks and one addition merge theta_next and
phi_next again. As in the upstream example, packed mode appends bias to the
model as an intercept coordinate. Each real sample row gets a public trailing
one, while padded rows remain all zero. The combined row
`[weights, bias, padding]` lets the forward pass and gradient update all
parameters together. Both complete NAG states therefore remain in one
ciphertext from one epoch to the next.

Packed sparse bootstrapping must retain at least two row blocks. The CLI
therefore raises the payload to 32 slots for a 16-wide model. It also reserves
two extra post-bootstrap levels: 12 instead of 10 for cubic, or 18 instead of
16 for Chebyshev. The default separate representation and its depths are
unchanged.

Simulated refresh re-encrypts the complete selected representation each epoch.
Neither refresh path resets momentum or replaces the previous phi with theta.
Real-mode metric/paired-benchmark decryptions never feed the training state.

Momentum defaults to 0.1 and must be finite in [0, 1). Zero momentum produces
the GD recurrence; separate mode can omit the unused previous state, while
packed mode retains the selected one-ciphertext layout. GD remains the default
so earlier lab measurements are reproducible. The CLI passes identical
settings to both trainers; the encrypted API also rejects a reference from a
different optimizer or momentum. CSV rows append `optimizer`, `momentum`
(zero for GD), `sigmoid`, and `nag_packing`.

## Framingham preprocessing

The port follows `heart_disease_data()`:

1. Remove every row containing any missing value.
2. Drop `education`, `currentSmoker`, `BPMeds`, `diabetes`, `diaBP`, and `BMI`.
3. Balance `TenYearCHD=0` and `TenYearCHD=1` by sampling each class with seed 73.
4. Keep nine features: `male`, `age`, `cigsPerDay`, `prevalentStroke`,
   `prevalentHyp`, `totChol`, `sysBP`, `heartRate`, and `glucose`.
5. Standardize each feature with the full balanced dataset's sample standard
   deviation (`ddof=1`).

This produces 1,114 records with nine features, matching the lab report.

## Refresh methods

Both methods start from the same encrypted data, model, epoch circuit, context,
keys, and learning rate.

### Simulated bootstrapping

After an encrypted epoch, the client decrypts the updated optimizer state and
immediately encrypts it again in the selected representation. Separate NAG
refreshes theta and the previous gradient-step model; packed NAG refreshes its
single state ciphertext. `refresh_seconds` measures both decryption and
re-encryption. This is the technique used by the
TenSEAL lab to simulate a refreshed modulus/noise budget.

### Real bootstrapping

OpenFHE 1.1.2 returns the original ciphertext when it still has at least as many
modulus towers as bootstrapping would produce. The real branch therefore waits
through the initial natural epochs until the consumed level is greater than the
post-bootstrap trigger. It then calls `EvalBootstrap` on each worn state
ciphertext: two for GD, four for separate NAG, or one for packed NAG. No
secret key is used for these refreshes. Once the model is at the shorter post-bootstrap chain, the
next epoch consumes enough
levels to make every subsequent bootstrap genuine.

At every genuine bootstrap point, the program also decrypts and re-encrypts a
copy of the exact same worn model and records
`paired_simulated_refresh_seconds`. That result is discarded; the real-mode
model continues only from `EvalBootstrap`. This isolates the two refresh costs
at the same epoch and level. A separate decryption of the bootstrapped model is
performed solely to report the lab's per-epoch accuracy and loss; its time is
`metric_decryption_seconds` and is not part of `refresh_seconds`.

## Timing definitions

- `homomorphic_seconds`: encrypted forward passes, gradient accumulation, and
  model update, including NAG extrapolation and packed state extraction/
  repacking when selected.
- `refresh_seconds`: decrypt+encrypt for simulated bootstrapping, or
  `EvalBootstrap` for real bootstrapping, across all retained optimizer state
  (two ciphertexts for GD, four for separate NAG with nonzero momentum, or one
  for packed NAG).
- `paired_simulated_refresh_seconds`: at a genuine real-bootstrap point, the
  cost of decrypting and re-encrypting a discarded copy of the same input,
  including both NAG states.
- `seconds_per_epoch`: `homomorphic_seconds + refresh_seconds`. Accuracy/loss
  calculation and real-mode metric-only decryption are excluded.
- `metric_decryption_seconds`: real-mode decryption used only to calculate the
  lab's epoch metrics.

The CSV also reports maximum consumed OpenFHE levels across all optimizer
ciphertexts before and after refresh, the selected `nag_packing`, and the maximum absolute coefficient
error relative to the matching plaintext epoch.

## OpenFHE configuration

For each dataset, the two refresh methods use one common context. It uses 2,048
data slots and at least 16 sparse model-bootstrap slots (32 for a 16-wide packed
NAG model), `FLEXIBLEAUTO`,
59-bit scaling moduli, a 60-bit first
modulus, `HYBRID` key switching, level budget `{3,3}`, and ring dimension 4096.
The degree-59 circuit reserves 16 levels after bootstrapping, for total
multiplicative depth 35 and a real-bootstrap trigger at consumed level 19.
The four extra towers beyond the 12 arithmetic levels are required for
OpenFHE 1.1.2 to bootstrap the level-30/31 GD/NAG ciphertexts without exhausting
the DCRT representation. The cubic option restores the main-branch reserve of
10 levels and total depth 29, with the same trigger at level 19. In
`CkksConfiguration`, `levelsAvailableAfterBootstrap = 0` selects these defaults;
a nonzero value explicitly overrides the reserve for custom experiments.
The small `HEStd_NotSet` ring is taken from OpenFHE 1.1.2's bootstrapping example
and makes no production security claim. These OpenFHE bootstrapping parameters
replace the TenSEAL-specific modulus-chain syntax; the default GD experiment
remains the same.

Packed row summation adds a masked reduction to the circuit, and NAG adds a
scalar momentum multiplication after the first epoch, so level consumption
and the first real-bootstrap epoch can differ from the old
per-sample implementation. Refresh is selected from the actual consumed
level, not forced by epoch number. Higher consumed level means less remaining
capacity; the log reports before/after refresh, not before/after training.
