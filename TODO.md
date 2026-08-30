# TODO

- [x] Add row-major sample packing adapted from the official OpenFHE example.
- [x] Add encrypted regression tests for padding, multiple blocks, and post-bootstrap continuation.
- [x] Improve README and document the packing/refresh design.
- [ ] Run repeated, controlled packed-vs-unpacked benchmarks; the four-epoch report compares new measurements with earlier runs, not a fresh controlled A/B experiment.
- [x] Add optional Nesterov accelerated gradient to plaintext and encrypted training, including state refresh and tests.
- [ ] Compare NAG and GD convergence and total runtime in repeated, controlled runs.
- [ ] Try different sigmoid approximations.
- [ ] Dockerize the project for reproducible builds and execution.
- [ ] merge feature/compare-GD-NAG
- [ ] merge feature/approximate-sigmoid