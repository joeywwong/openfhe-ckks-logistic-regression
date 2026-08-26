# Faithful lab-to-OpenFHE design

## Scope

This project ports the original TenSEAL experiment to OpenFHE. It does not
replace the lab's optimizer, split, preprocessing, sigmoid approximation,
metrics, or ciphertext-per-sample representation. The requested experiment
adds a second refresh branch so the lab's decrypt-and-encrypt refresh can be
measured against genuine CKKS bootstrapping.

## Behavior retained from the lab

| Part | Retained behavior |
|---|---|
| Datasets | Only `LogReg_sample_dataset.csv` and `framingham.csv` |
| Split | Shuffle, then first `floor(0.30*n)` records for test; seed 4 |
| Model | Binary logistic regression; zero-initialized weights and bias |
| Optimizer | Full-batch gradient descent |
| Default run | 100 epochs and learning rate 0.01 |
| Encrypted input | One encrypted feature vector and one encrypted label per training record |
| Model layout | Encrypted weights and encrypted bias are separate ciphertexts |
| Forward sigmoid | `0.5 + 0.197*z - 0.004*z^3` |
| Gradient | `(prediction-label)*x` and `(prediction-label)` for bias |
| Accuracy | Decrypt model, classify the plaintext test set at linear score zero |
| Loss | Exact logistic sigmoid and binary cross-entropy on plaintext training data |

The original lab standardizes the complete balanced Framingham dataset before
splitting it. Although fitting preprocessing before a split is normally changed
to avoid leakage, it is deliberately retained here because this is a
reproduction rather than a redesigned ML evaluation.

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

After an encrypted epoch, the client decrypts both updated model ciphertexts and
immediately encrypts the recovered weights and bias again. `refresh_seconds`
measures both decryption and re-encryption. This is the technique used by the
TenSEAL lab to simulate a refreshed modulus/noise budget.

### Real bootstrapping

OpenFHE 1.1.2 returns the original ciphertext when it still has at least as many
modulus towers as bootstrapping would produce. The real branch therefore waits
through the initial natural epochs until the consumed level is greater than the
post-bootstrap trigger. It then calls `EvalBootstrap` on the weight ciphertext
and bias ciphertext. No secret key is used for these two refreshes. Once the
model is at the shorter post-bootstrap chain, the next epoch consumes enough
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
  model update.
- `refresh_seconds`: decrypt+encrypt for simulated bootstrapping, or two
  `EvalBootstrap` calls for real bootstrapping.
- `paired_simulated_refresh_seconds`: at a genuine real-bootstrap point, the
  cost of decrypting and re-encrypting a discarded copy of the same input.
- `seconds_per_epoch`: `homomorphic_seconds + refresh_seconds`. Accuracy/loss
  calculation and real-mode metric-only decryption are excluded.
- `metric_decryption_seconds`: real-mode decryption used only to calculate the
  lab's epoch metrics.

The CSV also reports consumed OpenFHE levels before and after refresh and the
maximum absolute coefficient error relative to the matching plaintext epoch.

## OpenFHE configuration

The two experiments use one common context to isolate the refresh method. It
uses 16 sparse slots, `FLEXIBLEAUTO`, 59-bit scaling moduli, a 60-bit first
modulus, `HYBRID` key switching, level budget `{3,3}`, and ring dimension 4096.
The small `HEStd_NotSet` ring is taken from OpenFHE 1.1.2's bootstrapping example
and makes no production security claim. These OpenFHE bootstrapping parameters
replace the TenSEAL-specific modulus-chain syntax; the ML experiment remains
the same.
