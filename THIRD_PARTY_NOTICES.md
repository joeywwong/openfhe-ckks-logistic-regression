# Third-party notices

The row-major, zero-padded sample layout, repeated label/weight layout, and
`EvalSumCols`/`EvalSumRows` matrix-vector reduction approach in
`src/sample_packing.cpp` and `src/ckks_logistic_regression.cpp` are adapted from
[`openfheorg/openfhe-logreg-training-examples`](https://github.com/openfheorg/openfhe-logreg-training-examples),
commit `b9f38f4e8e6fc93ef5d2a3a5d880f80e72d0484d`, specifically `utils.cpp`,
`enc_matrix.h`, and the sparse model bootstrap pattern in `lr_nag.cpp`.
The local adaptation adds multiple ciphertext blocks and an explicit valid-row
mask while preserving this project's separate bias and default lab optimizer.
The optional fixed-momentum Nesterov update in `src/logistic_regression.cpp`
and `src/ckks_logistic_regression.cpp` is also adapted from `lr_nag.cpp` at the
same commit, including its first step without momentum. Both retained optimizer
states are refreshed using this project's separate weight/bias layout.

## Upstream BSD 2-Clause License

Copyright (c) 2023, Duality Technologies Inc.
All rights reserved.

Author TPOC: contact@openfhe.org

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
