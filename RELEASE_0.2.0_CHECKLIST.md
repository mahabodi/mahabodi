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

## RC build sizes (macOS arm64, built on the mini WITH the postgres feature)

Filled by scripts/rc_sizes output; 0.1.2 comparison values are the macOS x86_64 artifacts from dist/0.1.2
(different arch, so indicative only — the store + rustls growth is the point being tracked).
