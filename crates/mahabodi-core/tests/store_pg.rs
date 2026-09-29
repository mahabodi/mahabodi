//! The PostgreSQL store must rank like in-process memory: same stage and same top-k for the same documents and
//! queries (lexical cascade + spreading; no embedder here). Needs a server: MAHABODI_TEST_PG_DSN=postgres://...
#![cfg(feature = "postgres")]

use std::collections::HashMap;

use mahabodi_core::ingest::{ingest, Format};
use mahabodi_core::memory::Memory;
use mahabodi_core::query;
use mahabodi_core::store::pg::Store;

const DOCS: &[(&str, &str)] = &[
    ("# Barack Obama\n\nBarack Hussein Obama II is an American politician who served as the 44th president of the United States.", "pg1"),
    ("# Michelle Obama\n\nMichelle LaVaughn Robinson Obama is an American attorney and author who was first lady of the United States.", "pg2"),
    ("# Refund policy\n\nRefunds take five days. Escalate after two days to the billing team. Refunds over 500 need approval.", "kb"),
    ("# Billing\n\nInvoices are sent monthly. The billing team answers payment questions within a day.", "kb2"),
    ("## [ID: ATF_REF_01]\n**Action:** Approve_Refund\n**Input:** {Order_Id}\n**Context_Links:** [ATF_REF_02]\n## [ID: ATF_REF_02]\n**Action:** Notify_Customer\n**Input:** {Email}", "atf"),
    ("UUIDToken rotation uses B2B keys; the security team rotates them every quarter.", "sec"),
    ("# Kwun Tong Garden Estate\n\nA public housing estate; Lotus Tower was built in 1987 (Block 4).", "pg3"),
];

const QUERIES: &[&str] = &[
    "Obama", "American president", "refunds", "refund approval", "billing team", "Approve_Refund", "Notify",
    "uuid token", "rotating keys", "invoicing", "presidnet", "refnd", "Lotus Tower", "block 4", "united states",
    "garden estate", "nothing matches this zqxv",
];

#[test]
fn store_ranks_like_in_process() {
    let Ok(dsn) = std::env::var("MAHABODI_TEST_PG_DSN") else {
        eprintln!("skipped: MAHABODI_TEST_PG_DSN not set");
        return;
    };
    // in-process reference
    let mut m = Memory::new();
    m.ingest_many(&DOCS.iter().map(|(t, s)| (*t, Format::Auto, *s)).collect::<Vec<_>>());
    // store: the same documents, parsed by the same ingest, in two batches (cross-batch shared nodes)
    let ns = format!("t{}", std::process::id());
    let st = Store::open(&dsn, &ns, true).expect("open");
    for batch in DOCS.chunks(4) {
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
    st.build().expect("build");
    let mut diffs = Vec::new();
    for q in QUERIES {
        let a = query::query(m.graph(), m.index(), q, 5);
        let b = st.query(q, 5, None, 0.0).expect("query");
        let ia: Vec<&str> = if a.stage == query::Stage::Hub { vec![] } else { a.hits.iter().map(|h| h.id.as_str()).collect() };
        let ib: Vec<&str> = b.hits.iter().map(|h| h.id.as_str()).collect();
        if a.stage != b.stage || ia != ib || (a.term_coverage - b.term_coverage).abs() > 1e-9 || a.handoff != b.handoff {
            diffs.push(format!("{q:?}: in-process {:?} {ia:?} cov {:.3} handoff {} | store {:?} {ib:?} cov {:.3} handoff {}",
                               a.stage, a.term_coverage, a.handoff, b.stage, b.term_coverage, b.handoff));
        }
    }
    let mut c = postgres::Client::connect(&dsn, postgres::NoTls).unwrap();
    for t in ["node", "posting", "stem_posting", "link", "vocab", "vocab_gram", "meta"] {
        c.execute(&format!("DELETE FROM mahabodi_store.{t} WHERE ns = $1"), &[&ns]).unwrap();
    }
    assert!(diffs.is_empty(), "store differs from in-process:\n{}", diffs.join("\n"));
}
