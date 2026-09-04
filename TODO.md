# TODO

- [x] Add row-major sample packing adapted from the official OpenFHE example.
- [x] Add encrypted regression tests for padding, multiple blocks, and post-bootstrap continuation.
- [x] Improve README and document the packing/refresh design.
- [ ] Run repeated, controlled packed-vs-unpacked benchmarks; the four-epoch report compares new measurements with earlier runs, not a fresh controlled A/B experiment.
- [x] Add optional Nesterov accelerated gradient to plaintext and encrypted training, including state refresh and tests.
- [x] Add a repeatable, controlled GD-vs-NAG runner and aggregate convergence/runtime summaries.
- [x] Add selectable upstream-style one-ciphertext NAG packing while retaining the separate approach.
- [x] Add selectable cubic and degree-59 Chebyshev sigmoid approximations.
- [x] Document preliminary 20-epoch separate-vs-packed NAG measurements and their limitations.
- [ ] Run repeated, order-balanced separate-vs-packed NAG benchmarks and report mean and sample standard deviation.
- [ ] Compare NAG and GD convergence and total runtime in repeated, controlled runs.
- [ ] Collect and report full repeated 100-epoch GD-vs-NAG encrypted measurements.
- [ ] Migrate and revalidate the experiment on a current OpenFHE release.
- [ ] Dockerize the project for reproducible builds and execution.
