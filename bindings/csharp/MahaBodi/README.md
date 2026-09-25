# MahaBodi (.NET)

MahaBodi is a System-1 engine for AI agents, written in Rust. It pairs a fastmemory-style topology memory
(ingest, a density guard, a query cascade that always returns an answer or a handoff, and hybrid retrieval)
with typed decisions from [Laya](https://github.com/NandhaKishorM/laya) models run through ONNX Runtime.

```csharp
using MahaBodi;
using var b = new Bodi();
b.Ingest("Refunds take five days.", format: "text");
var r = b.Query("refunds");            // r["matched"], r["handoff"], r["stage"], r["hits"]
```

This package bundles the native library for linux-x64 and osx-x64; other platforms are not included yet.
Laya decisions also need ONNX Runtime (set `ORT_DYLIB_PATH`) and a Laya model exported to ONNX (see the
repository). Memory works without either.

Benchmarks, including where MahaBodi loses:
[BENCHMARKS.md](https://github.com/mahabodi/mahabodi/blob/main/BENCHMARKS.md). MIT license.
