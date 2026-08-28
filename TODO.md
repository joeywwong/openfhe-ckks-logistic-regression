# TODO

- [x] Add row-major sample packing adapted from the official OpenFHE example.
- [x] Add encrypted regression tests for padding, multiple blocks, and post-bootstrap continuation.
- [x] Improve README and document the packing/refresh design.
- [ ] Run repeated, controlled packed-vs-unpacked benchmarks; the four-epoch report compares new measurements with earlier runs, not a fresh controlled A/B experiment.
- [ ] Use Nesterov Accelerated Gradient, see if optimization will be accelerated.
- [ ] Try different sigmoid approximations.