#!/usr/bin/env bash
# Run the enterprise-store end-to-end test (test_store.py) against a PostgreSQL that has the
# age, vector and pg_trgm extensions (e.g. the image built from ./Dockerfile).
#
#   DSN="host=127.0.0.1 port=5432 user=mahabodi dbname=mb_test" ./deploy/postgres/test_store.sh
#
# It (re)creates the schema in the target database, then runs test_store.py for one fastmemory
# fixture and, if PARAGRAPHS is set to a JSONL file of {"text": ...}, for that corpus too.
# Needs: psql, python with psycopg>=3.2, MahaBodi's Python binding, models/minilm, ORT_DYLIB_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${DSN:?set DSN to a libpq connection string for an EMPTY test database}"
PY="${PYTHON:-python3}"

psql "$DSN" -v ON_ERROR_STOP=1 -q -c "DROP SCHEMA IF EXISTS mahabodi CASCADE" >/dev/null
psql "$DSN" -v ON_ERROR_STOP=1 -q -f "$ROOT/deploy/postgres/01_schema.sql" >/dev/null
echo "schema applied"

PYTHONPATH="$ROOT/bindings/python/python:${PYTHONPATH:-}" "$PY" "$ROOT/deploy/postgres/test_store.py" --dsn "$DSN" --ns e2e_robotics --fixture robotics
if [ -n "${PARAGRAPHS:-}" ]; then
  PYTHONPATH="$ROOT/bindings/python/python:${PYTHONPATH:-}" "$PY" "$ROOT/deploy/postgres/test_store.py" --dsn "$DSN" --ns e2e_corpus --paragraphs-file "$PARAGRAPHS"
fi
