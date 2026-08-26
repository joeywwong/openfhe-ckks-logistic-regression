# Security scope

This repository reproduces an educational lab experiment. It is not a
production cryptographic system, security audit, or medical prediction tool.

The benchmark deliberately uses OpenFHE's laptop-friendly CKKS bootstrapping
profile with `HEStd_NotSet` and ring dimension 4096. It makes no standard
security claim. Its purpose is to compare refresh behavior on one machine.

Simulated bootstrapping crosses the confidentiality boundary by decrypting and
re-encrypting the model after every epoch. It therefore requires the secret key
and is not non-interactive FHE continuation. Genuine `EvalBootstrap` refreshes
the ciphertext without the secret key. The executable contains both client and
evaluator responsibilities in one process for controlled measurement.

Never commit generated secret keys, sensitive datasets, credentials, or
private experiment output. The two committed datasets are the same educational
files already present in the referenced lab repository.
