# PostgreSQL 17 + Apache AGE + pgvector image for the MahaBodi memory store (Enterprise.md, P2/P3).
# The same versions were tested natively on macOS by deploy/postgres/test_store.sh
# (PostgreSQL 17.7, AGE 1.7.0, pgvector 0.8.0). This Dockerfile itself was NOT built here
# (no Docker daemon on the test machine); build and run test_store.sh against it before use.
FROM postgres:17-bookworm

ARG AGE_REF=PG17/v1.6.0-rc0
ARG PGVECTOR_REF=v0.8.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates flex bison postgresql-server-dev-17 \
 && git clone --depth 1 --branch "${AGE_REF}" https://github.com/apache/age.git /tmp/age \
 && make -C /tmp/age && make -C /tmp/age install \
 && git clone --depth 1 --branch "${PGVECTOR_REF}" https://github.com/pgvector/pgvector.git /tmp/pgvector \
 && make -C /tmp/pgvector OPTFLAGS="" && make -C /tmp/pgvector install \
 && rm -rf /tmp/age /tmp/pgvector \
 && apt-get purge -y build-essential git flex bison postgresql-server-dev-17 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# AGE must be preloaded; schema is applied on first start.
RUN echo "shared_preload_libraries = 'age'" >> /usr/share/postgresql/postgresql.conf.sample
COPY 01_schema.sql /docker-entrypoint-initdb.d/01_schema.sql
