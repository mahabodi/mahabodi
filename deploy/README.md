# deploy/ — enterprise deployment artefacts

These support [Enterprise.md](../Enterprise.md). Each file states how it was tested.

| Path | What | Tested |
|---|---|---|
| `postgres/01_schema.sql` | PostgreSQL + AGE + pgvector + pg_trgm schema; `create_namespace()` makes one AGE graph per namespace, with property/endpoint indexes | yes: PG 17.7, AGE 1.7.0, pgvector 0.8.0 |
| `postgres/test_store.py`, `postgres/test_store.sh` | end-to-end test: MahaBodi → store → hybrid search → Cypher neighbourhood → hydrate → MahaBodi answers | yes (results in Enterprise.md section 7) |
| `postgres/Dockerfile` | PG17 + AGE + pgvector image | not built (no Docker daemon on the test machine) |
| `sync/mahabodi_pg.py` | reference sync/search/hydration module (psycopg 3) | yes, via test_store.py |
| `models/triton/{laya,minilm}/config.pbtxt` | model-server configs for the Laya decision model and the MiniLM embedder | checked against the real ONNX files (`models/check_model_configs.py`); Triton not run |
| `compose/docker-compose.yml` | pattern P2: one store node + model server + workers | `docker compose config` only |
| `k8s/*.yaml` | patterns P3/P5: per-shard StatefulSet, namespace router, GPU model server + HPA | `kubeconform -strict` (k8s 1.30) only |
