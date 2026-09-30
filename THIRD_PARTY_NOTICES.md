# Third-party notices

MahaBodi's own code is released under the MIT License (see `LICENSE`). It builds on, and in
some distributions bundles, the following third-party work, which keeps its own license.

| Component | Used for | License | Source |
|---|---|---|---|
| Laya decision models (`convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`) and Laya's sequence format | the decision model (exported to ONNX) and the input format MahaBodi reproduces | Apache-2.0 | https://github.com/NandhaKishorM/laya, https://huggingface.co/convaiinnovations/laya |
| sentence-transformers/all-MiniLM-L6-v2 | dense text embedder (exported to ONNX) | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| fastmemory | ATF parser and clustering, used as a crate dependency (core features only); example inputs in `crates/mahabodi-core/tests/fixtures` | MIT (see `crates/mahabodi-core/tests/fixtures/FASTMEMORY_LICENSE`) | https://github.com/fastBuilderAI/memory |
| rust-louvain | Louvain clustering, used by fastmemory from its compiled library when available (bundled per platform in release packages) | BSD-3-Clause | https://crates.io/crates/rust-louvain |
| Rust, Python, Node, Java, .NET and Go dependencies | build and runtime | their own licenses, as declared in each package manifest | — |

If you redistribute exported model files (`models/*/model.onnx`), include the Apache-2.0 license
and any NOTICE of the upstream model with them, and credit the upstream authors.

## rust-louvain (bundled in release packages as a compiled library)

The full licence, reproduced as its clause 2 requires for binary distributions. FastBuilder.AI's name is not used to
endorse or promote MahaBodi (clause 3).

```text
BSD 3-Clause License

Copyright (c) 2026, FastBuilder.AI
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

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
```
