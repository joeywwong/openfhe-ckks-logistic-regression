# Faithful lab-to-OpenFHE design

## Scope

This project ports the original TenSEAL experiment to OpenFHE. By default it
retains the lab's optimizer, split, preprocessing, sigmoid approximation, and
metrics. Nesterov accelerated gradient is available as an optional optimizer.
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
| Forward sigmoid | `0.5 + 0.197*z - 0.004*z^3` |
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

For `d` features, width `R` is the next power of two at least `d`. A 2,048-slot
ciphertext holds `2048/R` sample rows. For two features and three samples:

```text
features: [x00,x01 | x10,x11 | x20,x21 | 0,0 | ...]
labels:   [y0, y0  | y1, y1  | y2, y2  | 0,0 | ...]
weights:  [w0, w1  | w0, w1  | w0, w1  | w0,w1 | ...]
bias:     [b,  b   | b,  b   | b,  b   | b,b | ...]
valid:    [1,  1   | 1,  1   | 1,  1   | 0,0 | ...]
```

1. Multiply features by repeated weights. `EvalSumCols` sums feature columns
   independently within each row and replicates each dot product over its row.
2. Add the encrypted bias, apply the unchanged cubic sigmoid, and subtract
   repeated encrypted labels.
3. Multiply errors by feature values. `EvalSumRows` sums sample rows for each
   feature column, producing a repeated weight-gradient row.
4. Mask errors by public row occupancy and use `EvalSumRows` for bias gradients.
   Zero feature padding alone is insufficient: padded rows would otherwise
   contribute `sigmoid(bias)` to the bias gradient.
5. Sum gradients from **all blocks**, divide by the actual training count
   (not padded rows), and make one update with learning rate 0.01.

This extends the upstream layout to multiple blocks without changing to
mini-batch SGD. Optional Nesterov acceleration uses the same full-batch
gradient. Pre-scaled data, Chebyshev sigmoid, and the upstream dataset are not
adopted. The upstream code is explicitly illustrative, not a performance
benchmark; measurements here refer only to this local adaptation.

LogReg uses width 2, 1,024 rows/block, one block and two input ciphertexts.
Framingham uses width 16, 128 rows/block, seven blocks and 14 input ciphertexts.
GD retains two separate encrypted model ciphertexts. NAG with nonzero
momentum retains four, as described below.

The model repeats with period `R`, which divides 16. Following the upstream
sparse-bootstrap pattern, a clone's slot metadata is set to 16 for
`EvalBootstrap` and back to 2,048 for matrix operations. This is valid only for
the periodic model, not for arbitrary sample ciphertexts. Unlike the upstream
combined theta/phi ciphertext, weights and bias are still refreshed separately
for each optimizer state.

Packing changes floating-point summation order, not the full-batch formula.
Integration tests compare every encrypted epoch with the matching plaintext
optimizer, covering partial blocks, both feature widths, multiple blocks, and
training after the first real bootstrap.

## Nesterov optimizer state

The optional `--optimizer nag` follows the fixed-momentum theta/phi recurrence
in upstream `lr_nag.cpp`, including its first epoch without momentum. The
plaintext and encrypted trainers both compute the gradient at theta, take an
unaccelerated step phi_next, then extrapolate theta_next from phi_next and the
previous phi. Both weights and bias participate; the reported/final model is
theta. See [the README](../README.md#nesterov-accelerated-gradient) for the formula.

Nonzero momentum requires two separate models (four ciphertexts). Both states
use the existing row-cloned layout and sparse bootstrap. Each worn ciphertext
is bootstrapped only when its consumed level exceeds the trigger; a less-worn
state can be retained without a no-op bootstrap. Simulated refresh re-encrypts
both states each epoch. Neither refresh path resets momentum or replaces the
previous phi with theta. Real-mode metric/paired-benchmark decryptions never
feed the training state.

Momentum defaults to 0.1 and must be finite in [0, 1). Zero momentum takes the
GD circuit without an extra encrypted state. GD remains the default so earlier
lab measurements are reproducible. The CLI passes identical settings to both
trainers; the encrypted API also rejects a reference from a different optimizer
or momentum. CSV rows append `optimizer` and `momentum` (zero for GD).

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

After an encrypted epoch, the client decrypts the updated model ciphertexts and
immediately encrypts the recovered weights and bias again. NAG also refreshes
the previous gradient-step model. `refresh_seconds` measures both decryption
and re-encryption. This is the technique used by the
TenSEAL lab to simulate a refreshed modulus/noise budget.

### Real bootstrapping

OpenFHE 1.1.2 returns the original ciphertext when it still has at least as many
modulus towers as bootstrapping would produce. The real branch therefore waits
through the initial natural epochs until the consumed level is greater than the
post-bootstrap trigger. It then calls `EvalBootstrap` on each worn weight/bias
ciphertext, including the NAG gradient-step state. No secret key is used for
these refreshes. Once the model is at the shorter post-bootstrap chain, the
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
  model update, including NAG extrapolation when selected.
- `refresh_seconds`: decrypt+encrypt for simulated bootstrapping, or
  `EvalBootstrap` for real bootstrapping, across all retained optimizer state
  (two ciphertexts for GD, four for NAG with nonzero momentum).
- `paired_simulated_refresh_seconds`: at a genuine real-bootstrap point, the
  cost of decrypting and re-encrypting a discarded copy of the same input,
  including both NAG states.
- `seconds_per_epoch`: `homomorphic_seconds + refresh_seconds`. Accuracy/loss
  calculation and real-mode metric-only decryption are excluded.
- `metric_decryption_seconds`: real-mode decryption used only to calculate the
  lab's epoch metrics.

The CSV also reports maximum consumed OpenFHE levels across all optimizer
ciphertexts before and after refresh and the maximum absolute coefficient
error relative to the matching plaintext epoch.

## OpenFHE configuration

For each dataset, the two refresh methods use one common context. It
uses 2,048 data slots and 16 sparse model-bootstrap slots, `FLEXIBLEAUTO`,
59-bit scaling moduli, a 60-bit first
modulus, `HYBRID` key switching, level budget `{3,3}`, and ring dimension 4096.
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
