# Gradient Descent versus Nesterov comparison

This project compares full-batch gradient descent (GD) with fixed-momentum
Nesterov accelerated gradient (NAG) while holding the dataset split,
zero-valued initialization, cubic sigmoid, learning rate, epoch count, CKKS
configuration, packing scheme, and refresh method constant.

## Controlled workflow

`scripts/run_gd_nag_comparison_wsl.sh` runs both optimizers in every repeat and
alternates their order. The default four repeats give each optimizer two
first-position runs, which gives a balanced result. Each optimizer uses a new process, but setup and input
encryption are excluded from the reported training time. The comparison keeps
the following separate:

- encrypted arithmetic;
- optimizer-state refresh;
- metric-only decryption;
- the discarded paired simulated-refresh measurement.

The primary convergence views are loss versus epoch and loss versus cumulative
training time. Final test accuracy is also reported, but it can remain constant
while loss improves. For every GD run, the summary finds the first NAG epoch
whose loss is no greater than GD's final loss. This provides epoch savings and
time-to-target speedup in addition to the fixed-epoch comparison.

Raw per-epoch measurements are retained so that convergence curves can be
plotted without relying on the aggregate summaries. Mean and sample standard
deviation are reported across repeats.

## Four-epoch smoke measurement

The checked-in result under `results/gd_nag_smoke_4epochs/` verifies the entire
workflow with:

- two repeats with alternating optimizer order;
- the 1,000-row LogReg sample dataset and its fixed seed-4 split;
- four epochs, learning rate `0.01`, and NAG momentum `0.1`;
- simulated refresh only.

| Metric, mean across two runs | GD | NAG |
|---|---:|---:|
| Final exact-sigmoid training loss | 0.656384 | 0.653542 |
| Final test accuracy | 0.996667 | 0.996667 |
| Encrypted training time, seconds | 8.443 | 9.420 |
| Homomorphic arithmetic, seconds | 7.963 | 8.410 |
| Optimizer-state refresh, seconds | 0.480 | 1.010 |
| Maximum consumed level | 6 | 7 |
| Maximum plaintext-model error | 5.03e-14 | 5.84e-14 |

At epoch four, NAG's loss was lower by `0.002842`, approximately 0.43% of GD's
final loss. NAG reached GD's final-loss target at epoch four, so this short run
showed no epoch saving. The mean encrypted training times were 8.443 s for GD and 9.420 s for NAG, so NAG took about 11.6% more time in this smoke experiment. The extra cost is
consistent with NAG retaining and refreshing both theta and phi model states.

This is a smoke measurement, not a performance conclusion: two timing samples
and four epochs are insufficient to establish acceleration. The intended study
uses both datasets, simulated and real bootstrapping, 100 epochs, and at least
four repeats with an even repeat count to balance optimizer order:

```bash
REPEATS=4 EPOCHS=100 DATASET=all REFRESH=both \
  ./scripts/run_gd_nag_comparison_wsl.sh
```

The full study should judge NAG on both time to a common loss and total runtime,
not only on its loss after the same number of epochs.
