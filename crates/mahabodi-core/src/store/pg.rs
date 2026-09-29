//! The PostgreSQL store (cargo feature `postgres`). Rows come from `store::derive`, so node ids, labels, texts,
//! links and weighted term/stem frequencies are the in-process graph's and index's. `query` ports
//! `query::query_with` stage by stage: exact BM25 (+ substring boost), substring, stem, fuzzy (MahaBodi's own
//! trigram Jaccard over the stored vocabulary), then spreading from non-passage nodes to linked passages, ranking
//! with the same tie order, and reciprocal-rank fusion with dense retrieval. Differences, documented in
//! DESIGN_STORE.md: no hub fallback (nothing matched is an empty handoff), f64 arithmetic in SQL vs f32 in process
//! (near-ties may order differently), and HNSW search is approximate once the vector index exists.

use std::collections::{BTreeMap, HashMap, HashSet};

use postgres::{Client, NoTls};
use r2d2_postgres::PostgresConnectionManager;

use crate::fastmemory::parser::Atf;
use crate::graph::{ClusterEngine, GraphInput, Level};
use crate::index::level_rank;
use crate::query::{Hit, QueryResult, Stage};
use crate::store::derive::{derive, NodeRow};
use crate::text;
use crate::error::{BodiError as Error, Result};

pub const SCHEMA_VERSION: i32 = 1;

const SCHEMA: &str = r#"
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS mahabodi_store;
CREATE TABLE IF NOT EXISTS mahabodi_store.meta (
    ns text PRIMARY KEY, schema_version int NOT NULL, dim int, n_docs bigint, avg_len double precision,
    built boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS mahabodi_store.node (
    ns text NOT NULL, id text NOT NULL, level smallint NOT NULL, label text NOT NULL, text text NOT NULL,
    len real NOT NULL, lower_id text NOT NULL, lower_label text NOT NULL, degree int NOT NULL DEFAULT 0,
    fdeg int NOT NULL DEFAULT 0, PRIMARY KEY (ns, id));
CREATE TABLE IF NOT EXISTS mahabodi_store.posting (ns text NOT NULL, term text NOT NULL, node_id text NOT NULL, tf real NOT NULL);
CREATE TABLE IF NOT EXISTS mahabodi_store.stem_posting (ns text NOT NULL, stem text NOT NULL, node_id text NOT NULL, tf real NOT NULL);
CREATE TABLE IF NOT EXISTS mahabodi_store.link (ns text NOT NULL, a text NOT NULL, b text NOT NULL, PRIMARY KEY (ns, a, b));
CREATE TABLE IF NOT EXISTS mahabodi_store.vocab (ns text NOT NULL, term text NOT NULL, df int NOT NULL, ngrams int NOT NULL, PRIMARY KEY (ns, term));
CREATE TABLE IF NOT EXISTS mahabodi_store.vocab_gram (ns text NOT NULL, gram text NOT NULL, term text NOT NULL);
"#;

type Pool = r2d2::Pool<PostgresConnectionManager<NoTls>>;

pub struct Store {
    pool: Pool,
    ns: String,
}

impl std::fmt::Debug for Store {
    // never print the DSN (it may carry a password)
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").field("ns", &self.ns).finish()
    }
}

fn pg(e: impl std::fmt::Display) -> Error {
    Error::Store(format!("postgres: {e}"))
}

/// A postgres error with the server's message (the plain Display is just "db error"). Never includes the DSN.
fn pge(e: postgres::Error) -> Error {
    match e.as_db_error() {
        Some(d) => Error::Store(format!("postgres: {} ({})", d.message(), d.code().code())),
        None => Error::Store(format!("postgres: {e}")),
    }
}

fn level_code(l: Level) -> i16 {
    match l {
        Level::Function => 0,
        Level::Data => 1,
        Level::Access => 2,
        Level::Event => 3,
        Level::Concept => 4,
    }
}

/// Passage text that gets a dense vector, exactly as `Memory::refresh_dense` builds it.
pub fn embed_text(n: &NodeRow) -> String {
    format!("{}\n{}", n.label.replace('_', " "), n.text)
}

impl Store {
    pub fn open(dsn: &str, ns: &str, create: bool) -> Result<Store> {
        if ns.is_empty() || !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(Error::Store("store namespace must be [A-Za-z0-9_]+".into()));
        }
        let cfg: postgres::Config = dsn.parse().map_err(pg)?;
        let pool = r2d2::Pool::builder().max_size(4).build(PostgresConnectionManager::new(cfg, NoTls)).map_err(pg)?;
        let s = Store { pool, ns: ns.to_string() };
        if create {
            let mut c = s.conn()?;
            c.batch_execute(SCHEMA).map_err(pge)?;
            c.execute("INSERT INTO mahabodi_store.meta (ns, schema_version) VALUES ($1, $2) ON CONFLICT (ns) DO NOTHING",
                      &[&s.ns, &SCHEMA_VERSION]).map_err(pge)?;
        }
        Ok(s)
    }

    fn conn(&self) -> Result<r2d2::PooledConnection<PostgresConnectionManager<NoTls>>> {
        self.pool.get().map_err(pg)
    }

    /// Write one batch (ATFs of one or more documents, already parsed by `ingest::ingest`). Newest wins per ATF id,
    /// as in-process: an existing passage with the same id is replaced with its postings, links and vector.
    /// `vectors` (optional) are unit vectors for the batch's Function nodes, in `derive` order.
    pub fn write_batch(&self, atfs: &[Atf], links: &[(String, String)], concepts: &[(String, String)],
                       texts: &HashMap<String, String>, embed: Option<&dyn Fn(&[String]) -> Result<Vec<Vec<f32>>>>) -> Result<usize> {
        let (nodes, edges) = derive(&GraphInput { atfs, links, concepts, texts, engine: ClusterEngine::Deterministic });
        let funcs: Vec<&NodeRow> = nodes.iter().filter(|n| n.level == Level::Function).collect();
        let vecs = match embed {
            Some(f) => Some(f(&funcs.iter().map(|n| embed_text(n)).collect::<Vec<_>>())?),
            None => None,
        };
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pge)?;
        let fids: Vec<String> = funcs.iter().map(|n| n.id.clone()).collect();
        // newest wins: drop earlier versions of these passages
        for t in ["posting", "stem_posting"] {
            tx.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1 AND node_id = ANY($2)"), &[&self.ns, &fids]).map_err(pge)?;
        }
        tx.execute("DELETE FROM mahabodi_store.link WHERE ns = $1 AND (a = ANY($2) OR b = ANY($2))", &[&self.ns, &fids]).map_err(pge)?;
        tx.execute("DELETE FROM mahabodi_store.node WHERE ns = $1 AND id = ANY($2)", &[&self.ns, &fids]).map_err(pge)?;
        let has_vec: bool = tx.query_one("SELECT to_regclass('mahabodi_store.vec') IS NOT NULL", &[]).map_err(pge)?.get(0);
        if has_vec {
            tx.execute("DELETE FROM mahabodi_store.vec WHERE ns = $1 AND node_id = ANY($2)", &[&self.ns, &fids]).map_err(pge)?;
        }
        // nodes: shared non-passage nodes (D_x, K_x) are inserted once; their postings only when newly inserted
        let (ids, lv, lab, txt, ln, lid, llab): (Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>) = nodes.iter().fold(
            Default::default(), |mut acc: (Vec<String>, Vec<i16>, Vec<String>, Vec<String>, Vec<f32>, Vec<String>, Vec<String>), n| {
                acc.0.push(n.id.clone()); acc.1.push(level_code(n.level)); acc.2.push(n.label.clone()); acc.3.push(n.text.clone());
                acc.4.push(n.len); acc.5.push(n.id.to_lowercase()); acc.6.push(n.label.to_lowercase()); acc });
        let rows = tx.query(
            "INSERT INTO mahabodi_store.node (ns, id, level, label, text, len, lower_id, lower_label)
             SELECT $1, * FROM unnest($2::text[], $3::int2[], $4::text[], $5::text[], $6::real[], $7::text[], $8::text[])
             ON CONFLICT (ns, id) DO NOTHING RETURNING id",
            &[&self.ns, &ids, &lv, &lab, &txt, &ln, &lid, &llab]).map_err(pge)?;
        let fresh: HashSet<String> = rows.iter().map(|r| r.get::<_, String>(0)).collect();
        let (mut pt, mut pn, mut pw, mut st, mut sn, mut sw) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for n in nodes.iter().filter(|n| fresh.contains(&n.id)) {
            for (t, w) in &n.tf { pt.push(t.clone()); pn.push(n.id.clone()); pw.push(*w); }
            for (t, w) in &n.stf { st.push(t.clone()); sn.push(n.id.clone()); sw.push(*w); }
        }
        tx.execute("INSERT INTO mahabodi_store.posting SELECT $1, * FROM unnest($2::text[], $3::text[], $4::real[])", &[&self.ns, &pt, &pn, &pw]).map_err(pge)?;
        tx.execute("INSERT INTO mahabodi_store.stem_posting SELECT $1, * FROM unnest($2::text[], $3::text[], $4::real[])", &[&self.ns, &st, &sn, &sw]).map_err(pge)?;
        let (la, lb): (Vec<String>, Vec<String>) = edges.into_iter().unzip();
        tx.execute("INSERT INTO mahabodi_store.link SELECT $1, * FROM unnest($2::text[], $3::text[]) ON CONFLICT DO NOTHING", &[&self.ns, &la, &lb]).map_err(pge)?;
        if let Some(vs) = vecs {
            let dim = vs.first().map(|v| v.len() as i32).unwrap_or(0);
            tx.batch_execute(&format!("CREATE TABLE IF NOT EXISTS mahabodi_store.vec (ns text NOT NULL, node_id text NOT NULL, embedding vector({dim}) NOT NULL, PRIMARY KEY (ns, node_id))")).map_err(pge)?;
            tx.execute("UPDATE mahabodi_store.meta SET dim = $2 WHERE ns = $1", &[&self.ns, &dim]).map_err(pge)?;
            let lits: Vec<String> = vs.iter().map(|v| format!("[{}]", v.iter().map(|x| format!("{x:.7}")).collect::<Vec<_>>().join(","))).collect();
            tx.execute("INSERT INTO mahabodi_store.vec SELECT $1, i, e::vector FROM unnest($2::text[], $3::text[]) AS u(i, e)", &[&self.ns, &fids, &lits]).map_err(pge)?;
        }
        tx.execute("UPDATE mahabodi_store.meta SET built = false WHERE ns = $1", &[&self.ns]).map_err(pge)?;
        tx.commit().map_err(pge)?;
        Ok(funcs.len())
    }

    /// Corpus statistics and indexes: degrees, BM25 N and average length, vocabulary document frequencies and
    /// MahaBodi's trigram sets (text::trigrams). Run after bulk loads, before querying.
    pub fn build(&self) -> Result<()> {
        let mut c = self.conn()?;
        c.batch_execute("
            CREATE INDEX IF NOT EXISTS posting_term ON mahabodi_store.posting (ns, term);
            CREATE INDEX IF NOT EXISTS stem_posting_stem ON mahabodi_store.stem_posting (ns, stem);
            CREATE INDEX IF NOT EXISTS link_b ON mahabodi_store.link (ns, b);
            CREATE INDEX IF NOT EXISTS vocab_gram_gram ON mahabodi_store.vocab_gram (ns, gram);
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            CREATE INDEX IF NOT EXISTS node_lower_id_trgm ON mahabodi_store.node USING gin (lower_id gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS node_lower_label_trgm ON mahabodi_store.node USING gin (lower_label gin_trgm_ops);").map_err(pg)?;
        let ns = &self.ns;
        c.execute("UPDATE mahabodi_store.node n SET degree = d.deg, fdeg = d.fdeg FROM (
                     SELECT x.id, count(*) AS deg, count(*) FILTER (WHERE o.level = 0) AS fdeg FROM (
                       SELECT a AS id, b AS other FROM mahabodi_store.link WHERE ns = $1
                       UNION ALL SELECT b, a FROM mahabodi_store.link WHERE ns = $1) x
                     JOIN mahabodi_store.node o ON o.ns = $1 AND o.id = x.other GROUP BY x.id) d
                   WHERE n.ns = $1 AND n.id = d.id", &[ns]).map_err(pge)?;
        c.execute("UPDATE mahabodi_store.meta m SET n_docs = s.n, avg_len = s.avg FROM (
                     SELECT count(*) AS n, avg(len)::float8 AS avg FROM mahabodi_store.node WHERE ns = $1) s WHERE m.ns = $1", &[ns]).map_err(pge)?;
        c.execute("DELETE FROM mahabodi_store.vocab WHERE ns = $1", &[ns]).map_err(pge)?;
        c.execute("DELETE FROM mahabodi_store.vocab_gram WHERE ns = $1", &[ns]).map_err(pge)?;
        let terms: Vec<(String, i64)> = c.query("SELECT term, count(*) FROM mahabodi_store.posting WHERE ns = $1 GROUP BY term", &[ns])
            .map_err(pg)?.iter().map(|r| (r.get(0), r.get(1))).collect();
        for chunk in terms.chunks(50_000) {
            let (mut t, mut df, mut ng, mut gg, mut gt) = (vec![], vec![], vec![], vec![], vec![]);
            for (term, n) in chunk {
                let grams = text::trigrams(term);
                t.push(term.clone()); df.push(*n as i32); ng.push(grams.len() as i32);
                for g in grams { gg.push(g); gt.push(term.clone()); }
            }
            c.execute("INSERT INTO mahabodi_store.vocab SELECT $1, * FROM unnest($2::text[], $3::int4[], $4::int4[])", &[ns, &t, &df, &ng]).map_err(pge)?;
            c.execute("INSERT INTO mahabodi_store.vocab_gram SELECT $1, * FROM unnest($2::text[], $3::text[])", &[ns, &gg, &gt]).map_err(pge)?;
        }
        c.execute("UPDATE mahabodi_store.meta SET built = true WHERE ns = $1", &[ns]).map_err(pge)?;
        Ok(())
    }

    fn stats(&self, c: &mut Client) -> Result<(f64, f64)> {
        let r = c.query_one("SELECT n_docs, avg_len, built FROM mahabodi_store.meta WHERE ns = $1", &[&self.ns]).map_err(pge)?;
        if !r.get::<_, bool>(2) {
            return Err(Error::Store("store not built: call store_build_index after loading".into()));
        }
        Ok((r.get::<_, Option<i64>>(0).unwrap_or(0) as f64, r.get::<_, Option<f64>>(1).unwrap_or(0.0)))
    }

    /// BM25 over `postings_table` for the given keys (each key's weight; repeated keys add again, as in process).
    fn bm25(&self, c: &mut Client, table: &str, col: &str, keys: &[(String, f64)], n: f64, avg: f64) -> Result<HashMap<String, f64>> {
        let (k1, b) = (1.2f64, 0.75f64);
        let mut out: HashMap<String, f64> = HashMap::new();
        for (key, weight) in keys {
            let rows = c.query(&format!(
                "SELECT p.node_id, p.tf, n.len, count(*) OVER () FROM mahabodi_store.{table} p
                 JOIN mahabodi_store.node n ON n.ns = p.ns AND n.id = p.node_id WHERE p.ns = $1 AND p.{col} = $2"),
                &[&self.ns, key]).map_err(pge)?;
            let df = rows.first().map(|r| r.get::<_, i64>(3) as f64).unwrap_or(0.0);
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for r in rows {
                let (tf, len) = (r.get::<_, f32>(1) as f64, r.get::<_, f32>(2) as f64);
                let norm = k1 * (1.0 - b + b * len / avg.max(1e-6));
                *out.entry(r.get(0)).or_default() += weight * idf * tf * (k1 + 1.0) / (tf + norm);
            }
        }
        Ok(out)
    }

    fn similar_terms(&self, c: &mut Client, t: &str, min_sim: f64, k: usize) -> Result<Vec<(String, f64)>> {
        let q: Vec<String> = text::trigrams(t).into_iter().collect();
        let rows = c.query("SELECT g.term, count(*)::int AS inter, v.ngrams FROM mahabodi_store.vocab_gram g
                            JOIN mahabodi_store.vocab v ON v.ns = g.ns AND v.term = g.term
                            WHERE g.ns = $1 AND g.gram = ANY($2) GROUP BY g.term, v.ngrams", &[&self.ns, &q]).map_err(pge)?;
        let mut out: Vec<(String, f64)> = rows.iter().map(|r| {
            let (inter, wl) = (r.get::<_, i32>(1) as f64, r.get::<_, i32>(2) as f64);
            (r.get::<_, String>(0), inter / (q.len() as f64 + wl - inter))
        }).filter(|(_, s)| *s >= min_sim).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.truncate(k);
        Ok(out)
    }

    /// `query::query_with` on the store. `qvec`: the query's unit vector, if an embedder is loaded.
    pub fn query(&self, q: &str, k: usize, qvec: Option<&[f32]>, min_similarity: f32) -> Result<QueryResult> {
        let k = k.max(1);
        let mut c = self.conn()?;
        let (n, avg) = self.stats(&mut c)?;
        let mut res = QueryResult { query: q.to_string(), matched: false, handoff: true, stage: Stage::EmptyMemory, confidence: 0.0,
                                    term_coverage: 0.0, hits: Vec::new(), blocks: Vec::new(), dense_similarity: None, corrections: Vec::new() };
        if n == 0.0 {
            return Ok(res);
        }
        let terms = text::terms(q);
        let mut uniq: Vec<String> = Vec::new();
        for t in &terms { if !uniq.contains(t) { uniq.push(t.clone()); } }
        let cover = |f: &dyn Fn(&str) -> bool| if terms.is_empty() { 0.0 } else { terms.iter().filter(|t| f(t)).count() as f64 / terms.len() as f64 };
        let mut scores = self.bm25(&mut c, "posting", "term", &uniq.iter().map(|t| (t.clone(), 1.0)).collect::<Vec<_>>(), n, avg)?;
        let ql = q.trim().to_lowercase();
        let substring: Vec<(String, String, String)> = if terms.is_empty() || q.trim().chars().count() < 3 { vec![] } else {
            let pat = format!("%{}%", ql.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
            c.query("SELECT id, lower_id, lower_label FROM mahabodi_store.node WHERE ns = $1 AND (lower_id LIKE $2 OR lower_label LIKE $2)",
                    &[&self.ns, &pat]).map_err(pge)?.iter().map(|r| (r.get(0), r.get(1), r.get(2))).collect()
        };
        let known: HashSet<String> = if uniq.is_empty() { HashSet::new() } else {
            c.query("SELECT term FROM mahabodi_store.vocab WHERE ns = $1 AND term = ANY($2)", &[&self.ns, &uniq]).map_err(pge)?
                .iter().map(|r| r.get(0)).collect()
        };
        let (mut stage, coverage) = if !scores.is_empty() {
            let top = scores.values().cloned().fold(0.0f64, f64::max);
            for (id, _, _) in &substring { *scores.entry(id.clone()).or_default() += 0.25 * top; }
            (Stage::Exact, cover(&|t| known.contains(t)))
        } else if !substring.is_empty() {
            scores = substring.iter().map(|(id, _, _)| (id.clone(), 1.0)).collect();
            (Stage::Substring, cover(&|t| substring.iter().any(|(_, li, ll)| li.contains(t) || ll.contains(t))))
        } else {
            let stems: Vec<(String, f64)> = uniq.iter().map(|t| (text::stem(t), 1.0)).collect();
            scores = self.bm25(&mut c, "stem_posting", "stem", &stems, n, avg)?;
            if !scores.is_empty() {
                let mut has = HashSet::new();
                for t in &uniq {
                    if !self.bm25(&mut c, "stem_posting", "stem", &[(text::stem(t), 1.0)], n, avg)?.is_empty() { has.insert(t.clone()); }
                }
                (Stage::Stem, cover(&|t| has.contains(t)))
            } else {
                let mut used = Vec::new();
                let mut keys = Vec::new();
                for t in &uniq {
                    for (w, sim) in self.similar_terms(&mut c, t, 0.4, 3)? {
                        keys.push((w.clone(), sim));
                        used.push((t.clone(), w, sim));
                    }
                }
                scores = self.bm25(&mut c, "posting", "term", &keys, n, avg)?;
                if !scores.is_empty() {
                    let hit: HashSet<&str> = used.iter().map(|(t, _, _)| t.as_str()).collect();
                    let cov = cover(&|t| hit.contains(t));
                    res.corrections = used;
                    (Stage::Fuzzy, cov)
                } else {
                    (Stage::Hub, 0.0) // no hub fallback in the store: an empty handoff unless dense answers
                }
            }
        };
        // spreading: a non-passage node passes 0.5 * s / sqrt(#linked passages) to each linked passage
        let ids: Vec<String> = scores.keys().cloned().collect();
        let meta: HashMap<String, (i16, i32, i32)> = if ids.is_empty() { HashMap::new() } else {
            c.query("SELECT id, level, degree, fdeg FROM mahabodi_store.node WHERE ns = $1 AND id = ANY($2)", &[&self.ns, &ids])
                .map_err(pg)?.iter().map(|r| (r.get(0), (r.get(1), r.get(2), r.get(3)))).collect()
        };
        if stage != Stage::Hub {
            let mut spread: BTreeMap<String, f64> = BTreeMap::new();
            let mut entries: Vec<(&String, &f64)> = scores.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let others: Vec<String> = entries.iter().filter(|(id, _)| meta.get(*id).map_or(false, |m| m.0 != 0)).map(|(id, _)| (*id).clone()).collect();
            let mut nbrs: HashMap<String, Vec<String>> = HashMap::new();
            if !others.is_empty() {
                for r in c.query("SELECT x.id, x.other FROM (SELECT a AS id, b AS other FROM mahabodi_store.link WHERE ns = $1 AND a = ANY($2)
                                  UNION ALL SELECT b, a FROM mahabodi_store.link WHERE ns = $1 AND b = ANY($2)) x
                                  JOIN mahabodi_store.node o ON o.ns = $1 AND o.id = x.other AND o.level = 0", &[&self.ns, &others]).map_err(pge)? {
                    nbrs.entry(r.get(0)).or_default().push(r.get(1));
                }
            }
            for (id, s) in entries {
                if meta.get(id).map_or(true, |m| m.0 == 0) {
                    *spread.entry(id.clone()).or_default() += s;
                } else {
                    let fs = nbrs.get(id).cloned().unwrap_or_default();
                    let share = 0.5 * s / (fs.len() as f64).sqrt().max(1.0);
                    for f in fs { *spread.entry(f).or_default() += share; }
                }
            }
            if !spread.is_empty() { scores = spread.into_iter().collect(); }
        }
        let ids: Vec<String> = scores.keys().cloned().collect();
        let info: HashMap<String, (i16, i32, String, String)> = if ids.is_empty() { HashMap::new() } else {
            c.query("SELECT id, level, degree, label, text FROM mahabodi_store.node WHERE ns = $1 AND id = ANY($2)", &[&self.ns, &ids])
                .map_err(pg)?.iter().map(|r| (r.get(0), (r.get(1), r.get(2), r.get(3), r.get(4)))).collect()
        };
        let lvl = |id: &str| info.get(id).map_or(Level::Concept, |m| code_level(m.0));
        let deg = |id: &str| info.get(id).map_or(0, |m| m.1);
        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            .then(level_rank(lvl(&a.0)).cmp(&level_rank(lvl(&b.0)))).then(deg(&b.0).cmp(&deg(&a.0))).then(a.0.cmp(&b.0)));
        let mut dense_ok = false;
        if let Some(v) = qvec {
            let lit = format!("[{}]", v.iter().map(|x| format!("{x:.7}")).collect::<Vec<_>>().join(","));
            let dr: Vec<(String, f64)> = c.query("SELECT node_id, -(embedding <#> $2::vector)::float8 AS sim FROM mahabodi_store.vec WHERE ns = $1
                                                  ORDER BY embedding <#> $2::vector, node_id LIMIT 50", &[&self.ns, &lit])
                .map_err(pg)?.iter().map(|r| (r.get(0), r.get(1))).collect();
            let top_sim = dr.first().map(|x| x.1).unwrap_or(0.0);
            res.dense_similarity = Some(top_sim);
            dense_ok = top_sim >= min_similarity as f64;
            if stage == Stage::Hub {
                if dense_ok { stage = Stage::Dense; ranked = dr; }
            } else {
                let mut fused: HashMap<String, f64> = HashMap::new();
                for (r, (i, _)) in ranked.iter().take(50).enumerate() { *fused.entry(i.clone()).or_default() += 1.0 / (60.0 + r as f64 + 1.0); }
                for (r, (i, _)) in dr.iter().enumerate() { *fused.entry(i.clone()).or_default() += 1.0 / (60.0 + r as f64 + 1.0); }
                ranked = fused.into_iter().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            }
        }
        if stage == Stage::Hub { ranked.clear(); }
        ranked.truncate(k);
        let need: Vec<String> = ranked.iter().map(|r| r.0.clone()).filter(|i| !info.contains_key(i)).collect();
        let mut info = info;
        if !need.is_empty() {
            for r in c.query("SELECT id, level, degree, label, text FROM mahabodi_store.node WHERE ns = $1 AND id = ANY($2)", &[&self.ns, &need]).map_err(pge)? {
                info.insert(r.get(0), (r.get(1), r.get(2), r.get(3), r.get(4)));
            }
        }
        let top = ranked.first().map(|r| r.1).unwrap_or(0.0).max(1e-9);
        for (id, s) in &ranked {
            let m = &info[id];
            res.hits.push(Hit { id: id.clone(), label: m.2.clone(), level: code_level(m.0), block: String::new(), score: s / top, text: m.3.clone() });
        }
        res.stage = if res.hits.is_empty() && stage == Stage::Hub { Stage::Hub } else { stage };
        res.matched = !matches!(res.stage, Stage::Hub | Stage::EmptyMemory);
        res.handoff = !res.matched || (coverage < 0.5 && !dense_ok);
        res.term_coverage = coverage;
        res.confidence = if res.stage == Stage::Dense {
            (res.stage.base_confidence() * res.dense_similarity.unwrap_or(0.0)).min(1.0)
        } else {
            (res.stage.base_confidence() * (0.5 + 0.5 * coverage.max(if dense_ok { 0.5 } else { 0.0 }))).min(1.0)
        };
        Ok(res)
    }

    pub fn stats_json(&self) -> Result<serde_json::Value> {
        let mut c = self.conn()?;
        let r = c.query_one("SELECT schema_version, dim, n_docs, avg_len, built FROM mahabodi_store.meta WHERE ns = $1", &[&self.ns]).map_err(pge)?;
        let passages: i64 = c.query_one("SELECT count(*) FROM mahabodi_store.node WHERE ns = $1 AND level = 0", &[&self.ns]).map_err(pge)?.get(0);
        Ok(serde_json::json!({"namespace": self.ns, "schema_version": r.get::<_, i32>(0), "dim": r.get::<_, Option<i32>>(1),
                              "nodes": r.get::<_, Option<i64>>(2), "avg_len": r.get::<_, Option<f64>>(3), "built": r.get::<_, bool>(4), "passages": passages}))
    }
}

fn code_level(c: i16) -> Level {
    match c {
        0 => Level::Function,
        1 => Level::Data,
        2 => Level::Access,
        3 => Level::Event,
        _ => Level::Concept,
    }
}
