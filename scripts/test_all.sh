#!/usr/bin/env bash
# Build and test MahaBodi in every language, against the real Laya model when models/laya-v2 exists.
# Prints one PASS/FAIL line per suite and exits non-zero if any suite fails.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$("$PY" -c "import onnxruntime,os,glob;print(glob.glob(os.path.join(os.path.dirname(onnxruntime.__file__),'capi','libonnxruntime.*'))[0])")}"
if [ -f "$ROOT/models/laya-v2/model.onnx" ]; then export BODI_LAYA_DIR="$ROOT/models/laya-v2"; else echo "NOTE: no Laya model - model tests SKIP"; fi
[ -f "$ROOT/models/minilm/model.onnx" ] && export BODI_EMBEDDER_DIR="$ROOT/models/minilm"
export PATH="$HOME/.dotnet:$HOME/.local/opt/apache-maven-3.9.9/bin:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1
LOG="$ROOT/target/test_all"; mkdir -p "$LOG"
fail=0
run() { # name, command...
  local name="$1"; shift
  if ( "$@" ) >"$LOG/$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name (see target/test_all/$name.log)"; fail=1; fi
}
# `cargo test` does not refresh the cdylibs the bindings load from target/release: build them
# explicitly, or Java/C#/Node would test a stale native library.
run native      cargo build --release -p mahabodi-ffi -p mahabodi-jni -p mahabodi-node
run rust        cargo test --workspace --release
run laya_parity cargo test -p mahabodi-core --release --test laya_parity -- --ignored
run python      bash -c "cd bindings/python && env -u CONDA_PREFIX VIRTUAL_ENV=$ROOT/.venv maturin develop --release -q && $PY -m unittest discover -s tests -v"
run node        bash -c "cd bindings/node && npm run build && npm test"
run java        bash -c "cd bindings/java && mvn -B test"
run csharp      bash -c "cd bindings/csharp/MahaBodi.Tests && dotnet test"
run go          bash -c "cd bindings/go && go test -count=1 -v ./..."
exit $fail
