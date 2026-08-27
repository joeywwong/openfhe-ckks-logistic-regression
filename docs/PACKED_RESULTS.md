# Packed CKKS logistic-regression results

Measured on 2026-08-27. This is a **four-epoch run on both complete
lab datasets**, with both refresh methods. The program's default remains 100
epochs; this report does not claim that 100 encrypted epochs were measured.

## What changed, and why the official approach fits

The [official OpenFHE example](https://github.com/openfheorg/openfhe-logreg-training-examples)
packs multiple sample rows into one CKKS ciphertext. Its
[`utils.cpp`](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/utils.cpp)
packs row-major features and repeats labels, while
[`enc_matrix.h`](https://github.com/openfheorg/openfhe-logreg-training-examples/blob/b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d/enc_matrix.h)
uses `EvalSumCols` for row-wise predictions and `EvalSumRows` for feature-wise
gradient sums.

These matrix operations express the same dot products and full-batch gradients
already used by this lab, so the layout is suitable. The adaptation adds
multiple blocks and an explicit valid-row mask for the separate bias gradient.
Every actual training sample contributes exactly once; padded rows do not
contribute, and the gradient denominator is the actual training count.

Only the ciphertext layout and its encrypted reductions were changed. The
datasets, preprocessing, seed-4 70/30 split, zero initialization, full-batch
gradient descent, learning rate 0.01, cubic sigmoid, and separate encrypted
weights and bias are retained. The upstream Nesterov optimizer, Chebyshev
sigmoid, and dataset were **not** imported. Its example is illustrative, not an
optimized benchmark; the measurements below are for this local adaptation.

See [DESIGN.md](DESIGN.md#packed-ciphertext-layout) for the layout and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for attribution.

## Packing and setup

Each block has one feature ciphertext and one label ciphertext. Counts below
exclude the two model ciphertexts, evaluation keys, and intermediate values.

| Dataset | Train / test | Features | Row width | Rows/block | Blocks | Input ciphertexts, old -> packed |
|---|---:|---:|---:|---:|---:|---:|
| LogReg sample | 700 / 300 | 2 | 2 | 1,024 | 1 | 1,400 -> 2 |
| Framingham | 780 / 334 | 9 | 16 | 128 | 7 | 1,560 -> 14 |

Framingham still has 1,114 records after the lab's original preprocessing.
All 700 and 780 training rows, respectively, were used in this benchmark;
the smaller subsets mentioned under tests are separate correctness checks.

| Dataset | Context and key setup (s) | Packing and input encryption (s) |
|---|---:|---:|
| LogReg sample | 5.174720 | 0.206536 |
| Framingham | 2.644040 | 0.320430 |

The unchanged demonstration ring has dimension 4,096 and multiplicative depth
29. Data now use all 2,048 slots; the periodically repeated weights and bias
use 16-slot sparse bootstrapping. The original `FLEXIBLEAUTO`, 59/60-bit
moduli, `HYBRID` key switching, and bootstrap level budget `{3,3}` remain.

## Four-epoch summary

Mean seconds/epoch is the arithmetic mean of the four recorded training
times, including each epoch's actual refresh when it occurs. It excludes
setup, input encryption, metric-only decryption, metric calculation, and the
discarded paired-refresh measurement.

| Dataset | Refresh | Mean s/epoch | Four-epoch training time (s) | Final test accuracy | Final training loss |
|---|---|---:|---:|---:|---:|
| LogReg sample | Simulated | 1.609 | 6.434 | 99.6667% | 0.6563837929 |
| LogReg sample | Real | 2.849 | 11.398 | 99.6667% | 0.6563831062 |
| Framingham | Simulated | 8.322 | 33.287 | 68.8623% | 0.6900177754 |
| Framingham | Real | 8.482 | 33.929 | 68.8623% | 0.6900177305 |

Accuracy is evaluated on the held-out plaintext test set after decrypting the
model. Loss is exact-sigmoid binary cross-entropy on the plaintext training
set; the encrypted training update itself still uses the lab's cubic sigmoid.

The unchanged plaintext reference ends at accuracy 99.6667% and loss
0.6563837929 for LogReg, and accuracy 68.8623% and loss 0.6900177754 for
Framingham. Both encrypted methods match the reference accuracy at every
recorded epoch. Real bootstrapping introduces small CKKS approximation error,
so loss and coefficients need not be bit-for-bit identical.

| Dataset | Refresh | Refresh events | Mean seconds per refresh event | Final max coefficient error |
|---|---|---:|---:|---:|
| LogReg sample | Simulated | 4 | 0.085560 | 8.646e-14 |
| LogReg sample | Real | 2 | 4.056956 | 7.271e-7 |
| Framingham | Simulated | 4 | 0.091832 | 8.839e-14 |
| Framingham | Real | 2 | 4.796196 | 3.618e-7 |

One **refresh event** covers both model ciphertexts: weights and bias.
For each dataset, real mode has two refresh events, which means four
individual `EvalBootstrap` calls. The mean refresh time above averages only
events that actually refresh, not the deferred epochs with zero refresh time.

## Paired simulated versus real refresh

At each real-refresh event, both techniques are timed on the same worn model.
The decrypt-and-encrypt result is discarded. Only the `EvalBootstrap` result
continues into the next real-mode epoch.

| Dataset | Epoch | Consumed level | Real refresh (s) | Paired simulated refresh (s) | Real / simulated |
|---|---:|---:|---:|---:|---:|
| LogReg sample | 3 | 20 | 4.034398 | 0.063886 | 63.2x |
| LogReg sample | 4 | 26 | 4.079514 | 0.043931 | 92.9x |
| Framingham | 3 | 20 | 5.477298 | 0.070008 | 78.2x |
| Framingham | 4 | 26 | 4.115094 | 0.053055 | 77.6x |

Simulated refresh is much cheaper here, but it requires the secret key and
exposes model values to its holder. Real bootstrapping refreshes the encrypted
model without decrypting it. Packing primarily reduces the cost and number of
encrypted training operations; it does not eliminate these two model refreshes.

This is a single local run. The paired measurement controls the input
ciphertexts and consumed level, but does not remove timing variability,
run-order, caching, or CPU scheduling effects.

## Per-epoch measurements

Times below are rounded for readability. The
[CSV](../results/benchmark_packed.csv) retains additional precision and
includes the plaintext reference, metric-decryption time, paired-refresh
time, and per-epoch model error.

### LogReg sample

| Refresh | Epoch | HE arithmetic (s) | Refresh (s) | Seconds/epoch | Test accuracy | Training loss | Consumed level |
|---|---:|---:|---:|---:|---:|---:|---|
| Simulated | 1 | 2.486 | 0.082 | 2.568 | 99.6667% | 0.6836835051 | 6 -> 0 |
| Simulated | 2 | 1.202 | 0.093 | 1.295 | 99.6667% | 0.6744039811 | 6 -> 0 |
| Simulated | 3 | 1.196 | 0.085 | 1.281 | 99.6667% | 0.6653052032 | 6 -> 0 |
| Simulated | 4 | 1.208 | 0.082 | 1.290 | 99.6667% | 0.6563837929 | 6 -> 0 |
| Real | 1 | 1.208 | 0.000 | 1.208 | 99.6667% | 0.6836835051 | 6 -> 6 |
| Real | 2 | 1.042 | 0.000 | 1.042 | 99.6667% | 0.6744039811 | 13 -> 13 |
| Real | 3 | 0.384 | 4.034 | 4.419 | 99.6667% | 0.6653051044 | 20 -> 19 |
| Real | 4 | 0.649 | 4.080 | 4.729 | 99.6667% | 0.6563831062 | 26 -> 19 |

### Framingham

| Refresh | Epoch | HE arithmetic (s) | Refresh (s) | Seconds/epoch | Test accuracy | Training loss | Consumed level |
|---|---:|---:|---:|---:|---:|---:|---|
| Simulated | 1 | 7.918 | 0.087 | 8.005 | 68.8623% | 0.6923548382 | 6 -> 0 |
| Simulated | 2 | 7.766 | 0.089 | 7.854 | 68.8623% | 0.6915691943 | 6 -> 0 |
| Simulated | 3 | 8.771 | 0.093 | 8.864 | 68.8623% | 0.6907901922 | 6 -> 0 |
| Simulated | 4 | 8.465 | 0.098 | 8.563 | 68.8623% | 0.6900177754 | 6 -> 0 |
| Real | 1 | 8.430 | 0.000 | 8.430 | 68.8623% | 0.6923548382 | 6 -> 6 |
| Real | 2 | 7.222 | 0.000 | 7.222 | 68.8623% | 0.6915691943 | 13 -> 13 |
| Real | 3 | 5.071 | 5.477 | 10.548 | 68.8623% | 0.6907901258 | 20 -> 19 |
| Real | 4 | 3.614 | 4.115 | 7.729 | 68.8623% | 0.6900177305 | 26 -> 19 |

### Reading the levels

These are **consumed** OpenFHE levels, not remaining levels. The arrow describes
the model immediately before and after refresh; it does not describe the start
and end of training.

- Simulated mode produces a fresh encryption each epoch: `6 -> 0`.
- Real mode defers refresh in epochs 1 and 2: `6 -> 6`, then `13 -> 13`.
- Real mode genuinely refreshes in epoch 3: `20 -> 19`.
- Epoch 4 trains from that bootstrapped model, then refreshes again: `26 -> 19`.

The packed reduction/masking circuit consumes levels differently from the
older per-sample circuit. Therefore the first genuine bootstrap now occurs
in epoch 3, not epoch 4. The trigger still uses the actual consumed level
(`level > 19`), not an epoch number. Returning to level 19 instead of zero
is expected for this real-bootstrap configuration.

## Comparison with the saved unpacked run

The existing [benchmark.csv](../results/benchmark.csv) was preserved byte for
byte. At the time of this update it contained a later Framingham-only run with
four plaintext, four simulated, and four real epochs; it is not the complete
raw data originally described in the historical [RESULTS.md](RESULTS.md).

| Framingham refresh | Saved unpacked mean s/epoch | Packed mean s/epoch | Observed ratio |
|---|---:|---:|---:|
| Simulated | 229.618 | 8.322 | 27.59x |
| Real | 229.898 | 8.482 | 27.10x |

These ratios compare **separate local runs**, not a fresh controlled A/B
experiment. The saved unpacked run had one real-refresh event in four epochs;
the packed run has two. CPU load, warm-up, and other system effects were not
controlled across the two runs. The figures show the observed improvement,
not a universal or statistically established speedup. No raw unpacked LogReg
comparison is inferred from this Framingham-only CSV.

Saved CSV SHA-256 at measurement time:

```text
BE376A58399E5E7F1B4BB52315C4BCCE244ACB20352C8CFB8009E13EA6D54071
```

## Timing definitions

- `homomorphic_seconds`: encrypted predictions, gradients, and model update.
- `refresh_seconds`: decryption plus re-encryption of weights and bias in
  simulated mode; both model bootstrap calls and their clone/slot preparation
  in real mode. Zero means refresh was deferred.
- `seconds_per_epoch`: exactly `homomorphic_seconds + refresh_seconds`.
- `paired_simulated_refresh_seconds`: discarded decrypt+encrypt comparison at
  the same real-refresh point; excluded from real `seconds_per_epoch`. In
  simulated rows it duplicates `refresh_seconds`, so do not add it again.
- `metric_decryption_seconds`: decryption of the real-mode model only to
  calculate accuracy, loss, and model error. It does not replace the encrypted
  model used in training. Simulated rows report zero because their refresh
  already provides a decrypted model.
- `max_plaintext_model_error`: maximum absolute difference across weights
  and bias versus the unchanged plaintext trainer at the same epoch.

The entire benchmark process took **94.38 seconds wall-clock**, including both
datasets, both methods, setup, packing/encryption, paired measurements, and
evaluation. Peak process resident memory was **764,100 KiB (746.19 MiB)**.
These process-wide values are not per-epoch metrics and are not a claim of
improvement over unmeasured historical memory usage.

## Build, correctness checks, and environment

The existing build script completed successfully. All **3/3 CTest tests passed**:

| Test | Scope | Time (s) |
|---|---|---:|
| `plaintext_tests` | Existing dataset and lab-training checks | 0.02 |
| `packing_tests` | Both full training sets, row/column padding, block boundaries, independent full-batch gradient comparison | 0.02 |
| `encrypted_packing_tests` | Four epochs, both methods, both feature widths, a multi-block tail, and post-bootstrap continuation | 45.85 |

The encrypted integration test uses 13 LogReg training rows and 129 Framingham
training rows from the same two lab datasets. Every test epoch must have
maximum coefficient error below `1e-5`, loss difference below `1e-6`, and the
same accuracy as its matching plaintext epoch. The full-dataset benchmark
above was then run separately.

The 24-row benchmark CSV was checked for four epochs per dataset/method,
finite metrics, timing sums, and agreement with the matching plaintext
epoch. Both dataset files and the plaintext/preprocessing implementation
were left unchanged.

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.4 LTS, WSL2; kernel 5.15.146.1-microsoft-standard-WSL2 |
| CPU | AMD Ryzen 7 7735U, 8 cores / 16 threads |
| Compiler | GNU C++ 11.4.0 |
| CMake used | 3.22.1 |
| OpenFHE | 1.1.2, 64-bit native backend |
| Build | Release; existing build-and-test script |
| Execution | One benchmark process, default threading; no CPU-affinity or repeated-trial controls |
| Security profile | Existing laptop demonstration, `HEStd_NotSet`; no production security claim |

The template's CMake 3.5.1 compatibility branch remains in place, but was
not executed with that exact historical CMake version.

## Reproduce

Run from the repository root in Linux/WSL:

```bash
./scripts/build_and_test_wsl.sh
EPOCHS=4 ./scripts/run_comparison_wsl.sh
```

The measured executable invocation was:

```bash
./build/openfhe_lab_compare --dataset all --refresh both --epochs 4 \
  --learning-rate 0.01 --output results/benchmark_packed.csv
```

The wrapper used `/usr/bin/time -v` and `tee` to retain process measurements
and the console transcript. Re-running the commands replaces the packed CSV;
choose `--output results/another_run.csv` (or the script's `OUTPUT_PATH`)
to keep a separate trial.

Artifacts:

- [Per-epoch benchmark CSV](../results/benchmark_packed.csv)
- [Console transcript](../results/packed_4epochs.log)
- [Process wall-time and memory measurement](../results/packed_runtime.txt)
- [Packing design](DESIGN.md#packed-ciphertext-layout)

Only four epochs were verified here. This is neither a convergence study nor
a production-secure FHE benchmark; long-run accuracy, noise behavior, and
repeatable timing distributions require separate measurements.
