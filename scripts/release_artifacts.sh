#!/usr/bin/env bash
# Build every release artifact for one version from ONE clean, pushed commit:
#   macOS x86_64 natives on this Mac; Linux x86_64 natives on the Ubuntu box via cargo-zigbuild at glibc 2.28
#   (asserted with objdump), then npm tgz, NuGet nupkg, Maven jar (natives bundled), Python wheels (abi3), and
#   crates dry-run packages. Writes dist/<version>/ with every artifact, BUILD_INFO.txt and SHA256SUMS.
# Nothing is published.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
UBUNTU_HOST="${UBUNTU_HOST:-pksingh@192.168.7.223}"
UBUNTU_REPO="${UBUNTU_REPO:-/media/sda/mahabodi-publish}"
LTARGET=x86_64-unknown-linux-gnu.2.28
if [ -n "$(git status --porcelain -- crates bindings Cargo.toml Cargo.lock | grep -v '^??')" ]; then
  echo "tracked changes under crates/, bindings/, Cargo.*: commit first" >&2; exit 1
fi
REV="$(git rev-parse HEAD)"
git fetch -q origin && [ "$(git rev-parse origin/main)" = "$REV" ] || { echo "HEAD $REV is not pushed as origin/main" >&2; exit 1; }
VER="$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys;print({p["name"]:p["version"] for p in json.load(sys.stdin)["packages"]}["mahabodi-core"])')"
D="$ROOT/dist/$VER"; rm -rf "$D"; mkdir -p "$D"
echo "== version $VER at $REV"

# ---- macOS natives
cargo build --release -q -p mahabodi-ffi -p mahabodi-node -p mahabodi-jni
cp target/release/libmahabodi_node.dylib bindings/node/mahabodi.darwin-x64.node

# ---- Linux natives on Ubuntu (same commit), glibc 2.28
ssh -o BatchMode=yes "$UBUNTU_HOST" "cd $UBUNTU_REPO && git checkout -q -- . && git fetch -q && git checkout -q $REV && \
  PATH=/media/sda/pubtools/bin:\$HOME/.cargo/bin:\$PATH cargo +1.94.0 zigbuild --release -q --target $LTARGET -p mahabodi-ffi -p mahabodi-node -p mahabodi-jni && \
  [ \"\$(git rev-parse HEAD)\" = $REV ]"
LR="$UBUNTU_REPO/target/x86_64-unknown-linux-gnu/release"
for f in libmahabodi.so libmahabodi_node.so libmahabodi_jni.so; do
  G="$(ssh -o BatchMode=yes "$UBUNTU_HOST" "objdump -T $LR/$f | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1")"
  [ "$G" = "GLIBC_2.28" ] || [ "$(printf '%s\nGLIBC_2.28\n' "$G" | sort -V | tail -1)" = "GLIBC_2.28" ] || { echo "$f needs $G > 2.28" >&2; exit 1; }
  echo "$f glibc floor $G"
done
scp -q "$UBUNTU_HOST:$LR/libmahabodi.so" target/release/libmahabodi.so
scp -q "$UBUNTU_HOST:$LR/libmahabodi_node.so" bindings/node/mahabodi.linux-x64.node

# ---- Java natives (the same script as before, same commit)
./scripts/java_natives.sh > /dev/null

# ---- Python wheels: Linux on Ubuntu (zig, manylinux_2_28), macOS here
ssh -o BatchMode=yes "$UBUNTU_HOST" "cd $UBUNTU_REPO/bindings/python && rm -rf /tmp/whl-$VER && \
  PATH=/media/sda/pubtools/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH maturin build --release -q --zig --compatibility manylinux_2_28 -o /tmp/whl-$VER"
scp -q "$UBUNTU_HOST:/tmp/whl-$VER/*.whl" "$D/"
( cd bindings/python && "$ROOT/.venv/bin/maturin" build --release -q -o "$D" 2>/dev/null || maturin build --release -q -o "$D" )

# ---- package
( cd bindings/node && npm pack -q --pack-destination "$D" > /dev/null )
( cd bindings/csharp/MahaBodi && "${DOTNET:-dotnet}" pack -c Release -o "$D" -v quiet > /dev/null )
( cd bindings/java && "${MVN:-mvn}" -B -q -DskipTests package && cp target/mahabodi-$VER.jar "$D/" )
cargo package -q -p mahabodi-core --allow-dirty --no-verify && cp target/package/mahabodi-core-$VER.crate "$D/"

{
  echo "version=$VER"; echo "git_commit=$REV"; echo "built_mac=$(hostname) $(rustc --version)"
  echo "built_linux=$(ssh -o BatchMode=yes "$UBUNTU_HOST" hostname) rustc 1.94.0 cargo-zigbuild $LTARGET"
  echo "java_natives: $(head -1 bindings/java/natives/BUILD_INFO.txt)"
} > "$D/BUILD_INFO.txt"
( cd "$D" && shasum -a 256 * > SHA256SUMS )
cat "$D/BUILD_INFO.txt"; cat "$D/SHA256SUMS"
