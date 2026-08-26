# Dataset provenance

This project intentionally uses only the two CSV files selected from the
original `fhe-ckks-lwe-encrypted-ml-lab` repository:

- `LogReg_sample_dataset.csv`: 1,000 two-feature, binary, linearly separable
  synthetic records.
- `framingham.csv`: the raw Framingham heart-study CSV used by the lab to
  predict `TenYearCHD`.

The files are copied without changing their values. Framingham filtering,
balancing, and standardization happen at runtime in `src/dataset.cpp` so the
steps remain reviewable and reproducible.

Source: <https://github.com/joeywwong/fhe-ckks-lwe-encrypted-ml-lab/tree/main/datasets>
