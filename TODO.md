# TODO

- [x] Add row-major sample packing adapted from the official OpenFHE example.
- [x] Add encrypted regression tests for padding, multiple blocks, and post-bootstrap continuation.
- [x] Improve README and document the packing/refresh design.
- [ ] Run repeated, controlled packed-vs-unpacked benchmarks; the four-epoch report compares new measurements with earlier runs, not a fresh controlled A/B experiment.
- [x] Add optional Nesterov accelerated gradient to plaintext and encrypted training, including state refresh and tests.
- [x] Add a repeatable, controlled GD-vs-NAG runner and aggregate convergence/runtime summaries.
- [ ] Collect and report full repeated 100-epoch GD-vs-NAG encrypted measurements.
- [ ] Try different sigmoid approximations.
- [ ] Dockerize the project for reproducible builds and execution.
