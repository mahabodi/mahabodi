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
    "escalations", "receipts", "overnight payments", "rotate keys safety", "settle",
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
    let dens = st.ensure_density(&mahabodi_core::density::DensityPolicy::default()).expect("density");
    let eng_dens = bodi.density();
    eprintln!("store density {dens}\nengine density {eng_dens}");
    assert!(dens["concepts_added"].as_u64().unwrap() > 0, "fixture must exercise density");
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
    for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "atf", "concept", "meta"] {
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
    for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "atf", "concept", "vec", "meta"] {
        let _ = c.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1"), &[&ns]);
    }
    assert!(diffs.is_empty(), "store (dense) differs from in-process:\n{}", diffs.join("\n"));
}
