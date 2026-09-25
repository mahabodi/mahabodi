# MahaBodi (Node.js)

MahaBodi is a System-1 engine for AI agents, written in Rust. It pairs a fastmemory-style topology memory
(ingest, a density guard, a query cascade that always returns an answer or a handoff, and hybrid retrieval)
with typed decisions from [Laya](https://github.com/NandhaKishorM/laya) models run through ONNX Runtime.

```js
const { Bodi } = require('mahabodi');
const b = new Bodi();
b.ingest('Refunds take five days.', { format: 'text' });
b.query('refunds');                 // { matched, handoff, stage, hits, ... }
// Laya decisions: b.loadLaya('models/laya-v2'); await b.decideAsync(state, questions)
```

Prebuilt native binaries: linux-x64 and darwin-x64 (Node 18+); other platforms build from source (see the
repository). Laya decisions also need ONNX Runtime (`ORT_DYLIB_PATH`) and a Laya model exported to ONNX;
memory works without either.

Benchmarks, including where MahaBodi loses:
[BENCHMARKS.md](https://github.com/mahabodi/mahabodi/blob/main/BENCHMARKS.md). MIT license.
