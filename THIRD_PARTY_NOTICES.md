# Third-party notices

MahaBodi's own code is released under the MIT License (see `LICENSE`). It builds on, and in
some distributions bundles, the following third-party work, which keeps its own license.

| Component | Used for | License | Source |
|---|---|---|---|
| Laya decision models (`convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`) and Laya's sequence format | the decision model (exported to ONNX) and the input format MahaBodi reproduces | Apache-2.0 | https://github.com/NandhaKishorM/laya, https://huggingface.co/convaiinnovations/laya |
| sentence-transformers/all-MiniLM-L6-v2 | dense text embedder (exported to ONNX) | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| fastmemory | ATF parser and Louvain clustering (and the algorithm ported in `louvain.rs`); example inputs in `crates/mahabodi-core/tests/fixtures` | MIT (see `crates/mahabodi-core/tests/fixtures/FASTMEMORY_LICENSE`) | https://github.com/fastBuilderAI/memory |
| Rust, Python, Node, Java, .NET and Go dependencies | build and runtime | their own licenses, as declared in each package manifest | — |

If you redistribute exported model files (`models/*/model.onnx`), include the Apache-2.0 license
and any NOTICE of the upstream model with them, and credit the upstream authors.
