# deploy/ — enterprise deployment artefacts

These support [Enterprise.md](../Enterprise.md). Each file states how it was tested.

| Path | What | Tested |
|---|---|---|
| `postgres/01_schema.sql` | PostgreSQL + AGE + pgvector + pg_trgm schema; `create_namespace()` makes one AGE graph per namespace, with property/endpoint indexes | yes: PG 17.7, AGE 1.7.0, pgvector 0.8.0 |
| `postgres/test_store.py`, `postgres/test_store.sh` | end-to-end test: MahaBodi → store → hybrid search → Cypher neighbourhood → hydrate → MahaBodi answers | yes (results in Enterprise.md section 7) |
| `postgres/Dockerfile` | PG17 + AGE + pgvector image | built and run on the Ubuntu box for the v1 5.9M entity-linking run (PostgreSQL 17, AGE `PG17/v1.6.0-rc0`, pgvector 0.8.0); the Mac mini runs the same versions natively (conda PostgreSQL 17.11) |
| `postgres/DESIGN_STORE.md` | design of the built-in PostgreSQL store (`store_*` engine calls, cargo feature `postgres`) | implemented on `main`, unreleased (planned for 0.2.0). Parity with the in-process engine is tested in `crates/mahabodi-core/tests/store_pg.rs`, against PostgreSQL 17.11 on the Mac mini. A Python round-trip and fork test is written (`bindings/python/tests/test_store_pg.py`) but **not yet run**. TLS: a plain server with `sslmode=prefer` is tested; a successful TLS handshake against a trusted-certificate server is **not yet tested**. The dev parity gate at 100K (`research/PREREG_SCALE_V2.md`) is pending. |
| `sync/mahabodi_pg.py` | reference sync/search/hydration module (psycopg 3) | yes, via test_store.py |
| `models/triton/{laya,minilm}/config.pbtxt` | model-server configs for the Laya decision model and the MiniLM embedder | checked against the real ONNX files (`models/check_model_configs.py`); Triton not run |
| `compose/docker-compose.yml` | pattern P2: one store node + model server + workers | `docker compose config` only |
| `k8s/*.yaml` | patterns P3/P5: per-shard StatefulSet, namespace router, GPU model server + HPA | `kubeconform -strict` (k8s 1.30) only |
