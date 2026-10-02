# 0.2.0 release candidate checklist (prepared 2026-10-03; release decision is the user's)

Branch `release-0.2.0`; versions bumped to 0.2.0 across the Cargo workspace, node package.json, csproj, pom.
Everything below names its evidence commit on main.

## Done and verified

| Item | Evidence |
|---|---|
| PostgreSQL store (schema 2, per-namespace vector tables/indexes, dense_tx, dense-mode NN join) | the PREREG_SCALE_V2 chain, reviewer-recomputed (eb2f1be) |
| Store parity with in-process at fixed code: hybrid 500/500 identical on dev; dev gate passes | db77abd, d8d84e7 |
| f64 avg_len fix confirmed as the bridge-divergence cause | 367941a |
| Auto-tag parser fix (prose lookalikes stay passages) | regression tests pass on main and on the fastmemory-dep build |
| TLS: rustls + `sslrootcert` (libpq semantics), DSN byte-identical pass-through, fail-fast store_open | 0f17af7, 4fb6b32, 63aac7d; two-sided scratch-cluster test 5/5 (b07aea5) |
| `sslmode=prefer` measured: no plaintext fallback on a failed handshake | store_tls_test.json, stated in CHANGELOG |
| Binding store tests, all five languages, against a real server | aceadff; all pass on the mini (Java via Temurin 21 + Maven 3.9.9 under ~/toolchains) |
| Non-skipping compiled-in checks (unreachable DSN) in every suite | aceadff + 52381a5 |
| Store compiled into every binding (opt-in `postgres` feature; node's npm build included) | aceadff, 52381a5 |
| Fork safety on macOS arm64 | bindings/python test, passing |
| fastmemory-dep swap, inline path: behaviour-identical on grounded dev decisions | 3ce990f (0 diffs of any kind) |

## Open before release

| Item | State |
|---|---|
| Linux fork test (test_store_pg.py on Ubuntu) | waits for clustering's LFR cells to finish |
| fastmemory-dep Ubuntu pair (dep_inline + dep_native, same machine) and the merge decision | waits for clustering; engine .so already on Ubuntu |
| Whether 0.2.0 ships from main (vendored) or from fastmemory-dep (crate) | decide after the Ubuntu pair |
| Default build of each published package: is `postgres` on by default or opt-in? | release scripts build WITH it (52381a5); the crates keep it opt-in for source users — confirm intent |
| Maven/Java distribution decision (jar with natives vs build-from-source note) | as 0.1.2 (jar packaged by release_artifacts.sh); confirm |
| Linux x86_64 artifacts + glibc 2.28 asserts (zigbuild on Ubuntu) | waits for clustering (heavy) |
| Package size growth vs 0.1.2 | macOS-arm64 RC numbers below; x86_64 at the real release build |
| Store docs page (deploy/postgres/DESIGN_STORE.md) read-through against shipped behaviour | pending |
| Re-gate parity at the release commit (optional per reviewer) | pending decision |
| Platforms: 0.1.2 shipped x86_64 only (Linux glibc 2.28, macOS x86_64); does 0.2.0 add macOS arm64 and/or Linux aarch64? Build + install-test every shipped platform; update the "prebuilt for …" lines | decide at release |
| Version strings in docs at the RELEASE commit, not before: README install/status lines, the one-pager footer "Alpha, version 0.1.2", the site, Go install `@v0.2.0` | release commit |
| Breaking change: the Auto-tag default needs a migration note in CHANGELOG + README ("pass format='entity_tags' to keep the old behaviour"); 0.x semver note (0.1 → 0.2 may break) | to write with the release |
| Publish order and tags: mahabodi-core → mahabodi-ffi → mahabodi; tags v0.2.0 and bindings/go/v0.2.0 at the release commit; registry verification against dist/0.2.0 (npm byte-identical, PyPI digests, crates checksums, NuGet by content) | at publish |
| Store release label: README presents the store as "new in 0.2.0; measured on one machine (Mac mini) to a 1M-page constructed pool", never production-proven | to write with the release |
| User decisions: keep or remove ~/toolchains on the mini; fastmemory 0.4.11 before or with 0.2.0 (the dep pins 0.4.10 either way) | user |

## RC build sizes (macOS arm64, built on the mini WITH the postgres feature)

(0.1.2 comparison values are macOS x86_64 from dist/0.1.2 — different arch, so indicative only; the store +
rustls growth is what is tracked. RC built at 95b1391 on the mini, postgres feature on.)

| Artifact | 0.1.2 (x86_64) | 0.2.0 RC (arm64) | Growth |
|---|---|---|---|
| Python wheel (compressed) | 2,409,469 B | 3,668,051 B | +52 % |
| Wheel's extension (uncompressed) | 6,205,288 B | — (in wheel) | — |
| libmahabodi.dylib (ffi) | (inside nupkg 4.5 MB) | 9,323,936 B | see note |
| libmahabodi_jni.dylib | (inside jar 4.5 MB) | 9,345,840 B | see note |
| libmahabodi_node.dylib | (inside tgz 4.6 MB) | 9,498,096 B | see note |

Note: 0.1.2 packages bundled TWO platform natives each (linux + mac x86_64) compressed; the RC numbers are one
raw uncompressed arm64 dylib, so package-level growth lands near the wheel's +52 %, driven by the store
(postgres/r2d2) and the TLS stack (rustls/ring/webpki). Exact per-package numbers come from
release_artifacts.sh's new size table at the real release build.
