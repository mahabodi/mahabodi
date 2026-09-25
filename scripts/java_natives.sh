#!/usr/bin/env bash
# Build the JNI native libraries bundled in the Java jar, both from the SAME clean git commit:
#   macos-x86_64 on this Mac, linux-x86_64 on the Ubuntu build box (UBUNTU_HOST, UBUNTU_REPO).
# Writes bindings/java/natives/<os>-<arch>/libmahabodi_jni.*, natives/BUILD_INFO.txt (commit, host,
# sha256 of each library) and bindings/java/meta/NOTICE (copy of THIRD_PARTY_NOTICES.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
UBUNTU_HOST="${UBUNTU_HOST:-pksingh@192.168.7.223}"
UBUNTU_REPO="${UBUNTU_REPO:-/media/sda/mahabodi-publish}"
if [ -n "$(git status --porcelain -- crates bindings/java Cargo.toml Cargo.lock)" ]; then
  echo "tree not clean under crates/, bindings/java, Cargo.*: commit first" >&2; exit 1
fi
REV="$(git rev-parse HEAD)"
git fetch -q origin && [ "$(git rev-parse origin/main)" = "$REV" ] || { echo "HEAD $REV is not pushed as origin/main" >&2; exit 1; }
N="$ROOT/bindings/java/natives"; rm -rf "$N" "$ROOT/bindings/java/meta"
mkdir -p "$N/macos-x86_64" "$N/linux-x86_64" "$ROOT/bindings/java/meta"
cargo build --release -q -p mahabodi-jni
cp target/release/libmahabodi_jni.dylib "$N/macos-x86_64/"
# Linux: stable toolchain (same as macOS) + zig, targeting glibc 2.28 so the library loads on older distros
LTARGET=x86_64-unknown-linux-gnu.2.28
ssh -o BatchMode=yes "$UBUNTU_HOST" "cd $UBUNTU_REPO && git checkout -q -- . && git fetch -q && git checkout -q $REV && \
  PATH=/media/sda/pubtools/bin:\$HOME/.cargo/bin:\$PATH cargo +1.94.0 zigbuild --release -q --target $LTARGET -p mahabodi-jni && git rev-parse HEAD && hostname"
scp -q "$UBUNTU_HOST:$UBUNTU_REPO/target/x86_64-unknown-linux-gnu/release/libmahabodi_jni.so" "$N/linux-x86_64/"
GLIBC="$(ssh -o BatchMode=yes "$UBUNTU_HOST" "objdump -T $UBUNTU_REPO/target/x86_64-unknown-linux-gnu/release/libmahabodi_jni.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1")"
[ "$GLIBC" = "GLIBC_2.28" ] || { echo "linux JNI needs $GLIBC, expected <= GLIBC_2.28" >&2; exit 1; }
UHOST="$(ssh -o BatchMode=yes "$UBUNTU_HOST" hostname)"; UREV="$(ssh -o BatchMode=yes "$UBUNTU_HOST" "cd $UBUNTU_REPO && git rev-parse HEAD")"
[ "$UREV" = "$REV" ] || { echo "Ubuntu built $UREV, expected $REV" >&2; exit 1; }
{
  echo "git_commit=$REV"
  echo "macos-x86_64/libmahabodi_jni.dylib built_on=$(hostname) sha256=$(shasum -a 256 "$N/macos-x86_64/libmahabodi_jni.dylib" | cut -d' ' -f1)"
  echo "linux-x86_64/libmahabodi_jni.so built_on=$UHOST sha256=$(shasum -a 256 "$N/linux-x86_64/libmahabodi_jni.so" | cut -d' ' -f1)"
  echo "rustc_macos=$(rustc --version)"
  echo "rustc_linux=$(ssh -o BatchMode=yes "$UBUNTU_HOST" '$HOME/.cargo/bin/rustc +1.94.0 --version') via cargo-zigbuild, target $LTARGET"
  echo "linux_glibc_floor=$GLIBC"
} > "$N/BUILD_INFO.txt"
cp THIRD_PARTY_NOTICES.md "$ROOT/bindings/java/meta/NOTICE"
cat "$N/BUILD_INFO.txt"
