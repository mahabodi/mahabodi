//! The PostgreSQL store must rank like in-process memory: same stage and same top-k for the same documents and
//! queries (lexical cascade + spreading; no embedder here). Needs a server: MAHABODI_TEST_PG_DSN=postgres://...
#![cfg(feature = "postgres")]

use std::collections::HashMap;

use mahabodi_core::ingest::{ingest, Format};
use mahabodi_core::store::pg::Store;

const DOCS: &[(&str, &str)] = &[
    ("# Barack Obama\n\nBarack Hussein Obama II is an American politician who served as the 44th president of the United States.", "pg1"),
    ("# Michelle Obama\n\nMichelle LaVaughn Robinson Obama is an American attorney and author who was first lady of the United States.", "pg2"),
    ("# Refund policy\n\nRefunds take five days. Escalate after two days to the billing team. Refunds over 500 need approval.", "kb"),
    ("# Billing\n\nInvoices are sent monthly. The billing team answers payment questions within a day.", "kb2"),
    ("## [ID: ATF_REF_01]\n**Action:** Approve_Refund\n**Input:** {Order_Id}\n**Context_Links:** [ATF_REF_02]\n## [ID: ATF_REF_02]\n**Action:** Notify_Customer\n**Input:** {Email}", "atf"),
    ("UUIDToken rotation uses B2B keys; the security team rotates them every quarter.", "sec"),
    ("# Kwun Tong Garden Estate\n\nA public housing estate; Lotus Tower was built in 1987 (Block 4).", "pg3"),
    // tie fixture: identical text under two sources
    ("# Tie page\n\nQuarterly audits check the ledger totals.", "tieA"),
    ("# Tie page\n\nQuarterly audits check the ledger totals.", "tieB"),
    // low-link passages (few data connections): density must add concept links, as in the engine
    ("refunds take five days and escalations take two.", "low1"),
    ("the ledger closes monthly after audits.", "low2"),
    ("keys rotate every quarter for safety.", "low3"),
    ("payments settle overnight in most cases.", "low4"),
    ("audits flag missing receipts early.", "low5"),
    ("Receipts.", "one1"),
    ("Escalations.", "one2"),
    ("Overnight.", "one3"),
    ("Ledgers.", "one4"),
];

const QUERIES: &[&str] = &[
    "Obama", "American president", "refunds", "refund approval", "billing team", "Approve_Refund", "Notify",
    "uuid token", "rotating keys", "invoicing", "presidnet", "refnd", "Lotus Tower", "block 4", "united states",
    "garden estate", "nothing matches this zqxv", "quarterly audit", "ledger", "tie page",
    "escalations", "receipts", "overnight payments", "rotate keys safety", "settle", "email", "close ticket", "Notify_Customer",
];

#[test]
fn store_ranks_like_in_process() {
    let Ok(dsn) = std::env::var("MAHABODI_TEST_PG_DSN") else {
        eprintln!("skipped: MAHABODI_TEST_PG_DSN not set");
        return;
    };
    // in-process reference: the ENGINE path the benchmarks used (one ingest_batch, with density), then Bodi::query
    let bodi = mahabodi_core::Bodi::new(mahabodi_core::BodiConfig::default()).expect("bodi");
    bodi.ingest_batch(&DOCS.iter().map(|(t, s)| (t.to_string(), Format::Auto, s.to_string())).collect::<Vec<_>>());
    // store: the same documents, parsed by the same ingest, in two batches (cross-batch shared nodes), then density
    let ns = format!("t{}", std::process::id());
    let st = Store::open(&dsn, &ns, true).expect("open");
    for batch in DOCS.chunks(5) {
        let (mut atfs, mut links, mut texts) = (Vec::new(), Vec::new(), HashMap::new());
        for (t, s) in batch {
            let mut n = 0;
            let g = ingest(t, Format::Auto, s, &mut n);
            atfs.extend(g.atfs);
            links.extend(g.links);
            texts.extend(g.texts);
        }
        st.write_batch(&atfs, &links, &[], &texts, None).expect("write");
    }
    // the engine runs density after every ingest_batch call; the store runs it when asked. Mirror the engine:
    // one density pass after the initial load (the engine's first call), then again after the re-ingest.
    let first = st.ensure_density(&mahabodi_core::density::DensityPolicy::default()).expect("density after load");
    assert!(first["concepts_added"].as_u64().unwrap() > 0, "fixture must exercise density: {first}");
    // re-ingest: an ATF id re-defined with a different action and input replaces the old one (newest wins) in both,
    // and nodes only the old version referenced (D_Email) must disappear from the store as from the in-process graph
    let redo = ("## [ID: ATF_REF_02]\n**Action:** Close_Ticket\n**Input:** {Ticket_Id}", "atf2");
    bodi.ingest_batch(&[(redo.0.to_string(), Format::Auto, redo.1.to_string())]);
    {
        let mut n = 0;
        let g = ingest(redo.0, Format::Auto, redo.1, &mut n);
        st.write_batch(&g.atfs, &g.links, &[], &g.texts, None).expect("rewrite");
    }
    let dens = st.ensure_density(&mahabodi_core::density::DensityPolicy::default()).expect("density");
    let eng_dens = bodi.density();
    eprintln!("store density {dens}\nengine density {eng_dens}");
    assert_eq!(dens["after"]["probe_recall"], eng_dens["probe_recall"]);
    assert_eq!(dens["after"]["min_links_per_function"], eng_dens["min_links_per_function"]);
    let mut diffs = Vec::new();
    for q in QUERIES {
        let a = bodi.query(q, 20);
        let b = st.query(q, 20, None, 0.0).expect("query");
        let stage_a = a["stage"].as_str().unwrap_or("").to_string();
        let stage_b = serde_json::to_value(b.stage).unwrap().as_str().unwrap().to_string();
        let ha: Vec<(String, f64)> = if stage_a == "hub" { vec![] } else {
            a["hits"].as_array().unwrap().iter().map(|h| (h["id"].as_str().unwrap().to_string(), h["score"].as_f64().unwrap())).collect() };
        let hb: Vec<(String, f64)> = b.hits.iter().map(|h| (h.id.clone(), h.score)).collect();
        let same = ha.len() == hb.len() && ha.iter().zip(&hb).all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-5);
        let cov_a = a["term_coverage"].as_f64().unwrap();
        if stage_a != stage_b || !same || (cov_a - b.term_coverage).abs() > 1e-9 || a["handoff"].as_bool().unwrap() != b.handoff {
            diffs.push(format!("{q:?}: engine {stage_a} {ha:?} cov {cov_a:.3} | store {stage_b} {hb:?} cov {:.3}", b.term_coverage));
        }
    }
    let mut c = postgres::Client::connect(&dsn, postgres::NoTls).unwrap();
    for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "atf", "concept", "ctxlink", "meta"] {
        c.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1"), &[&ns]).unwrap();
    }
    assert!(diffs.is_empty(), "store differs from in-process:\n{}", diffs.join("\n"));
}

/// With the MiniLM embedder loaded, store_query (per-passage vectors, exact search: no HNSW index) must rank like the
/// engine's in-process hybrid query. Needs MAHABODI_TEST_PG_DSN and MAHABODI_TEST_EMBEDDER_DIR (and ORT_DYLIB_PATH).
#[cfg(feature = "laya")]
#[test]
fn store_dense_ranks_like_in_process() {
    let (Ok(dsn), Ok(edir)) = (std::env::var("MAHABODI_TEST_PG_DSN"), std::env::var("MAHABODI_TEST_EMBEDDER_DIR")) else {
        eprintln!("skipped: MAHABODI_TEST_PG_DSN / MAHABODI_TEST_EMBEDDER_DIR not set");
        return;
    };
    let opts = mahabodi_core::system1::LoadOptions { ort_dylib: std::env::var("ORT_DYLIB_PATH").ok().map(std::path::PathBuf::from), ..Default::default() };
    let bodi = mahabodi_core::Bodi::new(mahabodi_core::BodiConfig::default()).expect("bodi");
    bodi.load_embedder(&edir, &opts).expect("embedder");
    let docs: Vec<serde_json::Value> = DOCS.iter().map(|(t, s)| serde_json::json!({"text": t, "source": s})).collect();
    bodi.call("ingest_batch", &serde_json::json!({"docs": docs})).expect("ingest");
    let ns = format!("d{}", std::process::id());
    bodi.call("store_open", &serde_json::json!({"dsn": dsn, "namespace": ns})).expect("open");
    for chunk in docs.chunks(5) {
        bodi.call("store_ingest_batch", &serde_json::json!({"docs": chunk})).expect("store ingest");
    }
    bodi.call("store_ensure_density", &serde_json::json!({})).expect("density");
    let mut diffs = Vec::new();
    for q in QUERIES.iter().chain(["what does the president do", "how long do refunds take", "security of API keys"].iter()) {
        let a = bodi.call("query", &serde_json::json!({"q": q, "k": 20})).unwrap();
        let b = bodi.call("store_query", &serde_json::json!({"q": q, "k": 20})).unwrap();
        let ids = |v: &serde_json::Value| -> Vec<(String, f64)> {
            if v["stage"] == "hub" { return vec![]; }
            v["hits"].as_array().unwrap().iter().map(|h| (h["id"].as_str().unwrap().to_string(), h["score"].as_f64().unwrap())).collect()
        };
        let (ha, hb) = (ids(&a), ids(&b));
        let same = ha.len() == hb.len() && ha.iter().zip(&hb).all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-4);
        let sim = |v: &serde_json::Value| v["dense_similarity"].as_f64().unwrap_or(-1.0);
        if a["stage"] != b["stage"] || !same || a["handoff"] != b["handoff"] || (sim(&a) - sim(&b)).abs() > 1e-4 {
            diffs.push(format!("{q:?}: engine {} {ha:?} sim {:.5} | store {} {hb:?} sim {:.5}", a["stage"], sim(&a), b["stage"], sim(&b)));
        }
    }
    let mut c = postgres::Client::connect(&dsn, postgres::NoTls).unwrap();
    let _ = c.batch_execute(&format!("DROP TABLE IF EXISTS mahabodi_store.\"vec_{ns}\""));
    for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "atf", "concept", "ctxlink", "meta"] {
        let _ = c.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1"), &[&ns]);
    }
    assert!(diffs.is_empty(), "store (dense) differs from in-process:\n{}", diffs.join("\n"));
}

/// A deterministic stand-in embedder (unit vectors from a hash of the text), so the vector-index tests need only a
/// server, not a model.
fn fake_embed(seed: u64) -> impl Fn(&[String]) -> mahabodi_core::Result<Vec<Vec<f32>>> {
    move |texts: &[String]| {
        Ok(texts.iter().map(|t| {
            let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
            let v: Vec<f32> = (0..8).map(|_| {
                for b in t.bytes() { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
                h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
                ((h >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            }).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / n).collect()
        }).collect())
    }
}

fn load_fake(st: &Store, docs: &[(String, String)], seed: u64) {
    let embed = fake_embed(seed);
    for batch in docs.chunks(50) {
        let (mut atfs, mut links, mut texts) = (Vec::new(), Vec::new(), HashMap::new());
        for (t, s) in batch {
            let mut n = 0;
            let g = ingest(t, Format::Auto, s, &mut n);
            atfs.extend(g.atfs);
            links.extend(g.links);
            texts.extend(g.texts);
        }
        st.write_batch(&atfs, &links, &[], &texts, Some(&embed)).expect("write");
    }
}

fn idx_scans(c: &mut postgres::Client, index: &str) -> i64 {
    c.batch_execute("SELECT pg_stat_clear_snapshot()").unwrap();
    c.query_opt("SELECT idx_scan FROM pg_stat_user_indexes WHERE schemaname = 'mahabodi_store' AND indexrelname = $1", &[&index])
        .unwrap().map_or(0, |r| r.get(0))
}

/// Vector indexes are per namespace: each namespace's approximate top-k (all lists probed) equals its exact top-k,
/// its queries scan its own index, a rebuild with other parameters replaces the old index and reports it, and a store
/// opened with another vector type is refused.
#[test]
fn store_vector_indexes_are_per_namespace() {
    use mahabodi_core::store::pg::QueryMode;
    let Ok(dsn) = std::env::var("MAHABODI_TEST_PG_DSN") else {
        eprintln!("skipped: MAHABODI_TEST_PG_DSN not set");
        return;
    };
    let (na, nb) = (format!("ia{}", std::process::id()), format!("ib{}", std::process::id()));
    let small: Vec<(String, String)> = DOCS.iter().map(|(t, s)| (t.to_string(), s.to_string())).collect();
    let big: Vec<(String, String)> = (0..600).map(|i| (format!("# Page {i}\n\nFiller passage number {i} about topic {}.", i % 37), format!("b{i}"))).collect();
    let a = Store::open(&dsn, &na, true).expect("open a");
    let b = Store::open(&dsn, &nb, true).expect("open b");
    load_fake(&a, &small, 1);
    load_fake(&b, &big, 2);
    let embed_a = fake_embed(1);
    let embed_b = fake_embed(2);
    let top = |st: &Store, q: &str, e: &dyn Fn(&[String]) -> mahabodi_core::Result<Vec<Vec<f32>>>| -> Vec<String> {
        let v = e(&[q.to_string()]).unwrap().remove(0);
        st.query_mode(q, 10, Some(&v), 0.0, QueryMode::Dense).expect("dense").hits.into_iter().map(|h| h.id).collect()
    };
    let qs = ["refunds", "president", "ledger audits", "keys", "topic 5", "filler"];
    // exact (no index yet), per namespace
    let exact_a: Vec<Vec<String>> = qs.iter().map(|q| top(&a, q, &embed_a)).collect();
    let exact_b: Vec<Vec<String>> = qs.iter().map(|q| top(&b, q, &embed_b)).collect();
    // each namespace's hits come only from its own vector table
    let mut c = postgres::Client::connect(&dsn, postgres::NoTls).unwrap();
    let ids = |c: &mut postgres::Client, ns: &str| -> std::collections::HashSet<String> {
        c.query(&format!("SELECT node_id FROM mahabodi_store.\"vec_{ns}\""), &[]).unwrap().iter().map(|r| r.get(0)).collect()
    };
    let (ids_a, ids_b) = (ids(&mut c, &na), ids(&mut c, &nb));
    assert!(ids_a.len() >= 10 && ids_b.len() >= 600 && ids_a.is_disjoint(&ids_b), "fixture: {} / {}", ids_a.len(), ids_b.len());
    assert!(exact_a.iter().flatten().all(|id| ids_a.contains(id)) && exact_b.iter().flatten().all(|id| ids_b.contains(id)),
            "a namespace returned another namespace's passage");
    // different lists per namespace; all lists probed, so IVFFlat must equal exact within the namespace
    assert!(a.build_ivfflat_index(2, 0, "64MB", 2).expect("ivf a").is_empty());
    assert!(b.build_ivfflat_index(20, 0, "64MB", 20).expect("ivf b").is_empty());
    let (ia, ib) = (format!("vec_{na}_ivfflat"), format!("vec_{nb}_ivfflat"));
    let (sa0, sb0) = (idx_scans(&mut c, &ia), idx_scans(&mut c, &ib));
    let approx_a: Vec<Vec<String>> = qs.iter().map(|q| top(&a, q, &embed_a)).collect();
    let approx_b: Vec<Vec<String>> = qs.iter().map(|q| top(&b, q, &embed_b)).collect();
    std::thread::sleep(std::time::Duration::from_secs(11)); // idle backends flush their statistics within 10 s
    let (sa1, sb1) = (idx_scans(&mut c, &ia), idx_scans(&mut c, &ib));
    assert_eq!(approx_a, exact_a, "namespace a: IVFFlat (all lists) differs from exact");
    assert_eq!(approx_b, exact_b, "namespace b: IVFFlat (all lists) differs from exact");
    assert!(sa1 - sa0 >= qs.len() as i64, "namespace a's queries did not scan its index ({sa0} -> {sa1})");
    assert!(sb1 - sb0 >= qs.len() as i64, "namespace b's queries did not scan its index ({sb0} -> {sb1})");
    // a rebuild with other parameters is explicit: the old definition is returned, the new one is in place
    let replaced = a.build_ivfflat_index(3, 0, "64MB", 3).expect("rebuild a");
    assert!(replaced.len() == 1 && replaced[0].contains("lists='2'"), "rebuild did not report the old index: {replaced:?}");
    let stats = a.stats_json().unwrap();
    let defs = stats["vector_indexes"].as_array().unwrap();
    assert!(defs.iter().any(|d| d.as_str().unwrap().contains("lists='3'")) && !defs.iter().any(|d| d.as_str().unwrap().contains("lists='2'")), "{stats}");
    assert!(b.stats_json().unwrap()["vector_indexes"].as_array().unwrap().iter().any(|d| d.as_str().unwrap().contains("lists='20'")), "b's index changed");
    // another vector type on an existing namespace is refused (build and write)
    let h = Store::open_with(&dsn, &na, false, "halfvec").expect("open halfvec");
    assert!(h.build_ivfflat_index(2, 0, "64MB", 2).is_err(), "halfvec build on a vector(8) namespace must fail");
    let mut n = 0;
    let g = ingest("# Extra\n\nOne more passage.", Format::Auto, "extra", &mut n);
    assert!(h.write_batch(&g.atfs, &g.links, &[], &g.texts, Some(&fake_embed(1))).is_err(), "halfvec write to a vector(8) namespace must fail");
    for ns in [&na, &nb] {
        c.batch_execute(&format!("DROP TABLE IF EXISTS mahabodi_store.\"vec_{ns}\"")).unwrap();
        for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "atf", "concept", "ctxlink", "meta"] {
            let _ = c.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1"), &[ns]);
        }
    }
}
