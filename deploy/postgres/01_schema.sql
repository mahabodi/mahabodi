-- MahaBodi enterprise memory store: PostgreSQL 16/17 + Apache AGE + pgvector + pg_trgm.
-- Reference schema (see Enterprise.md). Tested with PostgreSQL 17.7, AGE 1.7.0, pgvector 0.8.0
-- by deploy/postgres/test_store.sh.
--
-- Split of responsibilities:
--   relational tables  passage text, lexical index (tsvector), typo vocabulary (pg_trgm),
--                      dense vectors (pgvector HNSW). These are what the query cascade reads.
--   AGE graph          the fastmemory topology (Function / Data / Access / Event / Concept
--                      vertices and their edges), one graph per memory namespace, for
--                      traversal and neighbourhood extraction.
-- Every row carries `namespace` (a tenant or knowledge domain): the unit of isolation,
-- of sharding, and of hydration into an in-process MahaBodi working set.

CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
LOAD 'age';

CREATE SCHEMA IF NOT EXISTS mahabodi;
SET search_path = mahabodi, public;

CREATE TABLE IF NOT EXISTS namespace (
    name           text PRIMARY KEY CHECK (name ~ '^[a-z][a-z0-9_]{0,40}$'),
    graph          text NOT NULL UNIQUE,            -- AGE graph name, 'mb_' || name
    embedder       text NOT NULL,                   -- e.g. 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim  int  NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

-- One row per ATF (a function/passage node of the fastmemory topology).
-- Hash-partitioned by namespace so each partition stays a manageable size and a whole
-- namespace can be detached, moved or dropped. 16 partitions here; raise for large nodes.
CREATE TABLE IF NOT EXISTS atf (
    namespace         text   NOT NULL,
    id                text   NOT NULL,
    action            text   NOT NULL DEFAULT '',
    input             text   NOT NULL DEFAULT '',
    logic             text   NOT NULL DEFAULT '',
    access            text   NOT NULL DEFAULT '',
    events            text   NOT NULL DEFAULT '',
    data_connections  text[] NOT NULL DEFAULT '{}',
    body              text   NOT NULL DEFAULT '',   -- retrievable passage text (MahaBodi `texts[id]`)
    source            text,                         -- e.g. object-store URI of the original document
    content_hash      bytea,
    updated_at        timestamptz NOT NULL DEFAULT now(),
    tsv               tsvector GENERATED ALWAYS AS (
                          setweight(to_tsvector('simple', coalesce(action, '') || ' ' || id), 'A') ||
                          setweight(to_tsvector('english', body), 'B')) STORED,
    PRIMARY KEY (namespace, id)
) PARTITION BY HASH (namespace);

DO $$
BEGIN
    FOR i IN 0..15 LOOP
        EXECUTE format('CREATE TABLE IF NOT EXISTS mahabodi.atf_p%s PARTITION OF mahabodi.atf '
                       'FOR VALUES WITH (MODULUS 16, REMAINDER %s)', i, i);
    END LOOP;
END $$;

CREATE INDEX IF NOT EXISTS atf_tsv_gin ON atf USING gin (tsv);

-- Dense vectors (MahaBodi's hybrid stage). Dimension is fixed per embedder; 384 = MiniLM-L6.
-- A separate table so vectors can live on other storage, be rebuilt on an embedder change,
-- or be moved to a dedicated vector service without touching the graph.
CREATE TABLE IF NOT EXISTS atf_embedding (
    namespace  text NOT NULL,
    atf_id     text NOT NULL,
    model      text NOT NULL,
    embedding  vector(384) NOT NULL,
    PRIMARY KEY (namespace, atf_id, model),
    FOREIGN KEY (namespace, atf_id) REFERENCES atf (namespace, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS atf_embedding_hnsw ON atf_embedding USING hnsw (embedding vector_cosine_ops);

-- Vocabulary for typo correction (MahaBodi's fuzzy stage: trigram similarity over known terms).
CREATE TABLE IF NOT EXISTS vocab (
    namespace  text NOT NULL,
    term       text NOT NULL,
    doc_freq   int  NOT NULL DEFAULT 1,
    PRIMARY KEY (namespace, term)
);
CREATE INDEX IF NOT EXISTS vocab_term_trgm ON vocab USING gin (term gin_trgm_ops);

-- Create a namespace and its AGE graph with the fastmemory vertex/edge labels.
CREATE OR REPLACE FUNCTION create_namespace(ns text, embedder text, dim int) RETURNS text
LANGUAGE plpgsql AS $$
DECLARE g text := 'mb_' || ns; lbl text;
BEGIN
    INSERT INTO mahabodi.namespace (name, graph, embedder, embedding_dim) VALUES (ns, g, embedder, dim);
    PERFORM ag_catalog.create_graph(g::name);
    -- AGE's label functions take cstring arguments
    PERFORM ag_catalog.create_vlabel(g::cstring, l::cstring) FROM unnest(ARRAY['Function','Data','Access','Event','Concept']) AS l;
    PERFORM ag_catalog.create_elabel(g::cstring, l::cstring) FROM unnest(ARRAY['USES','GRANTS','EMITS','LINKS','MENTIONS']) AS l;
    -- AGE does not index vertex properties or edge endpoints. Without these, MATCH (f:Function {id: x})
    -- is a sequential scan (a `properties @> ...` filter) and bulk MERGE loads grow quadratically:
    -- measured here, 2,000 prose ATFs did not finish loading in 8 minutes without them.
    FOR lbl IN SELECT unnest(ARRAY['Function','Data','Access','Event','Concept']) LOOP
        EXECUTE format('CREATE INDEX %I ON %I.%I USING gin (properties)', lower(lbl) || '_props_gin', g, lbl);
    END LOOP;
    FOR lbl IN SELECT unnest(ARRAY['USES','GRANTS','EMITS','LINKS','MENTIONS']) LOOP
        EXECUTE format('CREATE INDEX %I ON %I.%I (start_id)', lower(lbl) || '_start', g, lbl);
        EXECUTE format('CREATE INDEX %I ON %I.%I (end_id)', lower(lbl) || '_end', g, lbl);
    END LOOP;
    RETURN g;
END $$;
