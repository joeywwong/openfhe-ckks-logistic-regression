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
Packing preserves the lab's data processing and cubic sigmoid. Full-batch
Nesterov accelerated gradient (NAG), adapted from the same example, is now an
optional optimizer; ordinary gradient descent (GD) remains the default.
The upstream Chebyshev sigmoid is not adopted.

## Lab behavior preserved by default

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

## Nesterov accelerated gradient (NAG)

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
- $g(\theta_t)$: full-batch gradient approximation using the cubic sigmoid.

The first epoch omits extrapolation because $\beta_0=0$.

Select `--optimizer nag`; `--momentum` defaults to `0.1`, must be finite in
`[0, 1)`, and `0` reduces to GD. The cubic sigmoid, full-batch averaging, and
learning rate are unchanged.

#### Convergence and encrypted-computation trade-offs
Compared with gradient descent (GD), nonzero-momentum NAG is intended to accelerate convergence and it may reach a target loss in fewer epochs (although acceleration is not guaranteed for this fixed-momentum implementation and the same learning rate as GD.). But in encrypted training, NAG has these tradeoffs:

- **Encrypted arithmetic:** After the first epoch, it performs additional encrypted arithmetic: two
  scalar multiplications, two additions, and two subtractions for weights and
  bias.
- **Model state:** It retains two encrypted model states, $\theta_t$ and $\phi_t$, each
  comprising weights and bias, versus GD's single model state.
- **Level consumption:** Its additional scalar multiplications may consume CKKS levels faster than GD
  and may therefore trigger real bootstrapping earlier.
- **Bootstrapping cost:** Under the current representation, refreshing both NAG states requires four
  bootstraps—weights and bias for each state—versus GD's two. This overhead is
  not inherent to NAG: the [official OpenFHE logistic-regression example](https://github.com/openfheorg/openfhe-logreg-training-examples#sparse-packing) packs both NAG states into one ciphertext and refreshes them with a single
  bootstrap. See [advanced CKKS bootstrapping](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-ckks-bootstrapping.cpp) for more information.

In a 100-epoch experiment, NAG showed faster loss reduction than GD under the
tested configuration. Larger momentum can also push scores outside the cubic
sigmoid's useful range.

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
The tests also check NAG against an independent velocity formulation, zero
momentum against GD, and encrypted NAG after real bootstrapping.
No CMake presets are needed. The workflow was verified with CMake 3.22.1;
the CMake 3.5.1 compatibility branch has not been executed locally.

## Run the comparison

### Controlled gradient descent versus Nesterov accelerated gradient comparison

Use the paired runner to compare convergence and runtime under identical data
splits, initialization, learning rate, epoch count, datasets, and refresh
methods. It preserves every raw per-epoch CSV and alternates whether GD or NAG
runs first, reducing systematic warm-cache and first-run bias. With the default
four repeats, each optimizer runs first twice, which gives a balanced result. For a short verification
experiment:

```bash
REPEATS=4 EPOCHS=4 DATASET=all REFRESH=both \
  ./scripts/run_gd_nag_comparison_wsl.sh
```

The defaults are `REPEATS=4`, `EPOCHS=100`, `MOMENTUM=0.1`,
`LEARNING_RATE=0.01`, `DATASET=all`, and `REFRESH=both`. A full run includes
real CKKS bootstrapping and can take a long time. `RESULT_DIR` selects an output
directory; otherwise a timestamped directory is created under `results/`.
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

`experiment_config.csv` records the controlled inputs. Reported total time is
the sum of encrypted arithmetic and optimizer-state refresh time; common
context setup and data encryption are intentionally excluded. Metric decryption
and the discarded paired-refresh measurement remain separate columns.

In the comparison files, positive `nag_final_loss_improvement` means NAG has
lower loss. Runtime ratios and speedups are `GD / NAG`, so values greater than
one favor NAG. Test accuracy should be interpreted alongside loss because its
discrete threshold can remain unchanged while optimization improves.

Existing raw result pairs can be summarized again without rerunning OpenFHE:

```bash
python3 scripts/summarize_gd_nag.py \
  --input-dir results/gd_nag_EXPERIMENT/raw \
  --output-dir results/gd_nag_EXPERIMENT
```

See [`docs/GD_NAG_COMPARISON.md`](docs/GD_NAG_COMPARISON.md) for the controlled
methodology and the checked-in two-repeat, four-epoch smoke measurement.

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
--output PATH
```

New measurements go to `results/benchmark_packed.csv` for GD or
`results/benchmark_nag.csv` for NAG. `OUTPUT_PATH` overrides that path when using
the script. CSV rows include `optimizer` and the effective `momentum` (zero for
GD). Existing result reports describe GD, not NAG. The earlier
`results/benchmark.csv` is left intact. See
[`docs/PACKED_RESULTS.md`](docs/PACKED_RESULTS.md) for the earlier packed GD run;
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
