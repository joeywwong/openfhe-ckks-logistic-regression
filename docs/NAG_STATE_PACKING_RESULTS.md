# One-ciphertext NAG state-packing results

This report documents a preliminary local comparison of the two encrypted
Nesterov accelerated gradient (NAG) state representations in this repository.
It is a correctness and engineering measurement, not a statistically
established or production-secure benchmark.

## Contribution

Separate mode retains four periodically refreshed ciphertexts:

- theta weights;
- theta bias;
- phi weights;
- phi bias.

Packed mode represents bias as an intercept coordinate and places complete
theta rows in even row blocks and complete phi rows in odd row blocks. An
alternating plaintext mask and a `+rowWidth` or `-rowWidth` rotation reconstruct
row-cloned theta and phi before the gradient. After the NAG update, masks merge
the two states into one ciphertext again. Real refresh therefore needs one
`EvalBootstrap` call instead of four.

This adapts the sparse NAG state layout from the
[official OpenFHE logistic-regression example](https://github.com/openfheorg/openfhe-logreg-training-examples)
while integrating it with this project's multiple data blocks, explicit bias,
selectable sigmoid circuits, two refresh methods, and per-epoch plaintext
validation. See [DESIGN.md](DESIGN.md#nesterov-optimizer-state) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for design details and
attribution.

## Measurement scope

The checked-in CSVs contain one 20-epoch run per layout using:

- full-batch NAG with momentum `0.1` and learning rate `0.01`;
- the cubic training sigmoid;
- genuine OpenFHE CKKS bootstrapping;
- the complete 700-row LogReg and 780-row Framingham training sets;
- the same seed-4 split, zero initialization, preprocessing, and metrics;
- 18 real refresh events per dataset and layout.

Training time is `homomorphic_seconds + refresh_seconds`. Context/key setup,
input encryption, metric-only decryption, and the discarded paired simulated
refresh measurement are excluded. Both layouts use the same definition.

The raw sources are [nag_separate.csv](../results/nag_separate.csv) and
[nag_packed.csv](../results/nag_packed.csv). No independent environment capture
or repeated/order-balanced trials were stored with these two files. The timing
results must therefore be described as preliminary observations from this
machine, not general performance claims.

## Twenty-epoch timing result

| Dataset | Layout | HE arithmetic (s) | Real refresh (s) | Total training (s) | Refreshes |
|---|---|---:|---:|---:|---:|
| Framingham | Separate | 85.127 | 146.154 | 231.281 | 18 |
| Framingham | Packed | 54.105 | 31.357 | 85.462 | 18 |
| LogReg sample | Separate | 12.606 | 124.570 | 137.176 | 18 |
| LogReg sample | Packed | 15.345 | 30.111 | 45.456 | 18 |

| Dataset | Total-time reduction | Total-time speedup | Refresh-time reduction | Refresh-time speedup |
|---|---:|---:|---:|---:|
| Framingham | 63.0% | 2.71x | 78.5% | 4.66x |
| LogReg sample | 66.9% | 3.02x | 75.8% | 4.14x |

The main benefit comes from replacing four model-state bootstraps with one.
Packing also reduced Framingham homomorphic arithmetic by 36.4%. On the
two-feature LogReg dataset, however, packed arithmetic increased by 21.7%:
adding the intercept raises its row width from two to four and changes the
training input from one block/two ciphertexts to two blocks/four ciphertexts.
The refresh saving still reduced total time substantially. This dataset-dependent
trade-off is why arithmetic and refresh time are reported separately.

## Numerical agreement

| Dataset | Layout | Final test accuracy | Final training loss | Maximum coefficient error over 20 epochs |
|---|---|---:|---:|---:|
| Framingham | Separate | 0.6856287425 | 0.6771098372 | 1.030e-6 |
| Framingham | Packed | 0.6856287425 | 0.6771097513 | 5.902e-7 |
| LogReg sample | Separate | 0.9966666667 | 0.5217960044 | 6.923e-7 |
| LogReg sample | Packed | 0.9966666667 | 0.5217960552 | 8.074e-7 |

The layouts have identical displayed final accuracy. Their final losses differ
by `8.59e-8` on Framingham and `5.08e-8` on LogReg. Every encrypted epoch is
compared with the matching plaintext optimizer; small differences are expected
from approximate CKKS arithmetic and bootstrapping.

## Correctness coverage

The encrypted integration tests exercise separate and packed NAG with both the
cubic and degree-59 Chebyshev sigmoid circuits. They cover simulated and real
refresh for packed state, verify that real bootstrapping restores levels, and
continue training after the first bootstrap. The 129-row Framingham case crosses
the 128-row block boundary and checks a heavily padded final block.

The coefficient check requires maximum parameter error below `1e-5`. Exact
logistic loss is 1-Lipschitz in the linear score, so its allowed difference is
derived from that epoch's measured coefficient error and the dataset's mean
augmented feature norm. This avoids treating randomized CKKS approximation
variation as a fixed decimal-precision guarantee.

On 2026-09-04, all five registered CTest tests passed in the current local
build. The cubic encrypted integration suite, which had previously exposed the
fixed-tolerance problem, also passed a second standalone run after the
sensitivity-bound change.

## Reproduction

Build and test first:

```bash
./scripts/build_and_test_wsl.sh
```

Generate comparable single-run files without overwriting the checked-in data:

```bash
./build/openfhe_lab_compare \
  --dataset all --refresh real --epochs 20 --learning-rate 0.01 \
  --optimizer nag --momentum 0.1 --nag-packing separate --sigmoid cubic \
  --output results/nag_separate_new.csv

./build/openfhe_lab_compare \
  --dataset all --refresh real --epochs 20 --learning-rate 0.01 \
  --optimizer nag --momentum 0.1 --nag-packing packed --sigmoid cubic \
  --output results/nag_packed_new.csv
```

For a controlled timing claim, run at least four paired repeats, alternate
which layout runs first, preserve every raw CSV, and report the mean and sample
standard deviation for arithmetic, refresh, and total training time. Record the
exact commit, CPU, OS, compiler, CMake, OpenFHE build, and run order with the
results.

## Security and interpretation limits

The default ring dimension 4,096 context uses `HEStd_NotSet` for a laptop-scale
demonstration. It makes no production security claim. Real `EvalBootstrap`
continues from encrypted state without exposing model values to the secret-key
holder; simulated refresh decrypts them. See [SECURITY.md](../SECURITY.md).

The measurements isolate the effect of optimizer-state representation in this
implementation. They do not establish that NAG is faster than GD, that the
speedup transfers to other hardware or parameter sets, or that the educational
configuration is suitable for clinical deployment.
