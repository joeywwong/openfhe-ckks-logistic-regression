# TODO

- [x] Add row-major sample packing adapted from the official OpenFHE example.
- [x] Add encrypted regression tests for padding, multiple blocks, and post-bootstrap continuation.
- [x] Improve README and document the packing/refresh design.
- [ ] Run repeated, controlled packed-vs-unpacked benchmarks; the four-epoch report compares new measurements with earlier runs, not a fresh controlled A/B experiment.
- [x] Add optional Nesterov accelerated gradient to plaintext and encrypted training, including state refresh and tests.
- [x] Add a repeatable, controlled GD-vs-NAG runner and aggregate convergence/runtime summaries.
- [ ] Compare NAG and GD convergence and total runtime in repeated, controlled runs.
- [ ] Collect and report full repeated 100-epoch GD-vs-NAG encrypted measurements.
- [ ] Try different sigmoid approximations.
- [ ] Dockerize the project for reproducible builds and execution.
- [x] merge feature/compare-GD-NAG
- [x] merge feature/approximate-sigmoid. Users can choose between sigmoid approximations (cubic polynomial used in the lab, or chebyshev approximation used by OpenFHE logistic regression example repo)
