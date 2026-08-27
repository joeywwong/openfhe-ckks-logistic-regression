# Measured comparison

> Historical **unpacked** implementation. For the sample-packing update and
> new four-epoch measurements, see [PACKED_RESULTS.md](PACKED_RESULTS.md).

These historical values were measured on the complete lab datasets, without
subsampling or sample packing. The current `results/benchmark.csv` contains a
later Framingham-only four-epoch run, not the complete raw data for the original
tables below. That CSV was preserved during the packing update; its comparison
with the new run is documented in [PACKED_RESULTS.md](PACKED_RESULTS.md).

The executable retains the lab default of 100 epochs. The checked-in benchmark
uses four real-mode epochs—the first point at which OpenFHE 1.1.2 genuinely
restores levels—and a clean simulated-mode epoch for each dataset. A complete
100-epoch encrypted run was not represented as measured because the retained
one-ciphertext-per-record design takes several minutes per epoch.

## Measurement environment

| Item | Value |
|---|---|
| Date | 2026-08-26 |
| OS | Ubuntu 22.04.4 LTS under WSL2 |
| CPU | AMD Ryzen 7 7735U, 8 cores / 16 threads |
| WSL memory | 7.4 GiB plus 2.0 GiB swap |
| Compiler | GCC 11.4.0 |
| CMake | 3.22.1 |
| OpenFHE | 1.1.2, 64-bit native backend |
| Build | Release, two build jobs |
| CKKS demo context | ring dimension 4096, 16 sparse slots, depth 29 |

The clean results below were taken without another OpenFHE benchmark running at
the same time. Setup and dataset-encryption time are intentionally excluded from
`seconds/epoch`.

## Dataset preparation

| Dataset | Total | Features | Train | Test |
|---|---:|---:|---:|---:|
| `LogReg_sample_dataset` | 1,000 | 2 | 700 | 300 |
| `framingham` after lab preprocessing | 1,114 | 9 | 780 | 334 |

The split is the lab's shuffled 70/30 split with seed 4. Framingham uses the
lab's missing-row removal, six-column removal, class balancing with seed 73,
and standardization before splitting.

## Direct refresh comparison

At real-mode epoch 4, both refresh techniques were timed on the same worn model
ciphertexts. The decrypt-and-encrypt result was discarded, and only the genuine
`EvalBootstrap` result continued as the real-mode model.

| Dataset | Consumed level | Simulated refresh (s) | Real refresh (s) | Real / simulated | Level after real | Max model error |
|---|---:|---:|---:|---:|---:|---:|
| LogReg sample | 23 | 0.0548 | 4.1592 | 75.9× | 19 | 4.97e-8 |
| Framingham | 23 | 0.0567 | 3.6830 | 65.0× | 19 | 6.85e-8 |

Each refresh includes two model ciphertexts: weights and bias. Real
bootstrapping restored the consumed level from 23 to 19 without using the
secret key. Simulated bootstrapping returned a new level-0 encryption but
required decrypting the model.

## Simulated-bootstrapping epoch

This is the lab-faithful branch: decrypt and re-encrypt after every epoch.

| Dataset | Epoch | HE arithmetic (s) | Refresh (s) | Seconds/epoch | Accuracy | Loss | Level |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogReg sample | 1 | 228.251 | 0.110 | 228.361 | 0.9967 | 0.683684 | 5 → 0 |
| Framingham | 1 | 276.571 | 0.088 | 276.659 | 0.6886 | 0.692355 | 5 → 0 |

The maximum encrypted-versus-plaintext model error was below `3e-15` in both
rows, before any genuine bootstrap approximation was introduced.

## Real-bootstrapping epochs

OpenFHE 1.1.2 returns the original ciphertext when bootstrapping would not add
modulus towers. The implementation therefore defers the operation through the
first three natural epochs and performs the first genuine refresh at epoch 4.

### LogReg sample

| Epoch | HE arithmetic (s) | Real refresh (s) | Seconds/epoch | Accuracy | Loss | Model error | Level |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 249.407 | 0 | 249.407 | 0.9967 | 0.683684 | 9.15e-16 | 5 → 5 |
| 2 | 273.231 | 0 | 273.231 | 0.9967 | 0.674404 | 4.08e-15 | 11 → 11 |
| 3 | 208.695 | 0 | 208.695 | 0.9967 | 0.665305 | 3.38e-15 | 17 → 17 |
| 4 | 149.353 | 4.159 | 153.513 | 0.9967 | 0.656384 | 4.97e-8 | 23 → 19 |

### Framingham

| Epoch | HE arithmetic (s) | Real refresh (s) | Seconds/epoch | Accuracy | Loss | Model error | Level |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 266.189 | 0 | 266.189 | 0.6886 | 0.692355 | 3.79e-15 | 5 → 5 |
| 2 | 273.701 | 0 | 273.701 | 0.6886 | 0.691569 | 1.85e-15 | 11 → 11 |
| 3 | 268.186 | 0 | 268.186 | 0.6886 | 0.690790 | 2.29e-15 | 17 → 17 |
| 4 | 179.324 | 3.683 | 183.007 | 0.6886 | 0.690018 | 6.85e-8 | 23 → 19 |

## Interpretation

- Decrypt-and-encrypt is roughly 65–76 times faster than genuine bootstrapping
  for this two-ciphertext model, but it requires the secret key and exposes the
  model between epochs.
- Real bootstrapping adds approximately 2–3% to the measured epoch in which it
  occurs. Sequential encrypted training dominates the runtime.
- Accuracy is unchanged by CKKS evaluation and bootstrapping at the displayed
  precision. The first real bootstrap increases model error from roughly
  `1e-15` to roughly `1e-8`, still far below a level that changes predictions or
  reported loss materially.
- LogReg accuracy is already high after the first epoch because the synthetic
  classes are linearly separable. Framingham accuracy remains about 0.689 over
  these first four small learning-rate updates, while exact-sigmoid training
  loss decreases each epoch.
- The high seconds/epoch values are a direct consequence of retaining the lab's
  one-feature-ciphertext and one-label-ciphertext per record. No minibatching,
  sample packing, subsampling, or parallel rewrite was added.

## Reproduction commands

```bash
./scripts/build_and_test_wsl.sh
./build/openfhe_lab_compare --dataset logreg --refresh real --epochs 4 \
  --output results/logreg_real_4epochs.csv
./build/openfhe_lab_compare --dataset framingham --refresh real --epochs 4 \
  --output results/framingham_real_4epochs.csv
```

For the original 100-epoch default with both methods:

```bash
./scripts/run_comparison_wsl.sh
```
