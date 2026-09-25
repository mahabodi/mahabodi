//! Memory, density and query-cascade behaviour on real fastmemory inputs and on
//! deliberately sparse/degenerate ones. Fixtures in tests/fixtures are fastmemory's
//! own example inputs (github.com/fastBuilderAI/memory, MIT, see FASTMEMORY_LICENSE).

use mahabodi_core::density::{self, DensityPolicy};
use mahabodi_core::graph::fastmemory_edges;
use mahabodi_core::ingest::Format;
use mahabodi_core::memory::Memory;
use mahabodi_core::query::{self, fastmemory_search, Stage};

const FIXTURES: &[&str] = &["business_analytics", "email_analysis", "health_science", "robotics", "world_events"];

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/{name}.md", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// What plain fastmemory (its Rust parser + Louvain + search) does with the same input.
fn fastmemory_alone(input: &str, q: &str) -> usize {
    let atfs = mahabodi_core::fastmemory::parser::parse_markdown(input);
    let json = mahabodi_core::fastmemory::cluster::run_louvain_inline(&fastmemory_edges(&atfs), &atfs);
    fastmemory_search(&json, q).as_array().map_or(0, |a| a.len())
}

#[test]
fn fastmemory_alone_returns_nothing_on_its_own_examples_bodi_finds_them() {
    for name in FIXTURES {
        let input = fixture(name);
        let mut m = Memory::new();
        let rep = m.ingest(&input, Format::Auto, name);
        assert!(rep.atfs_added >= 10, "{name}: {rep:?}");
        // take a real term from the first ATF's logic line and query it
        let first_logic = input.lines().find(|l| l.starts_with("**Logic:**")).unwrap();
        let term = mahabodi_core::text::terms(first_logic).into_iter().max_by_key(|t| t.len()).unwrap();
        assert_eq!(fastmemory_alone(&input, &term), 0, "{name}: fastmemory alone unexpectedly found {term}");
        let r = query::query(m.graph(), m.index(), &term, 5);
        assert!(r.matched, "{name}: {term} -> {:?}", r.stage);
        assert_eq!(r.stage, Stage::Exact);
    }
}

#[test]
fn density_guard_passes_on_every_fixture() {
    let p = DensityPolicy::default();
    for name in FIXTURES {
        let mut m = Memory::new();
        m.ingest(&fixture(name), Format::Auto, name);
        let out = density::ensure(&mut m, &p);
        assert!(out.after.passes, "{name}: {:#?}", out.after);
        assert_eq!(out.after.isolated_functions, 0);
        assert!(out.after.probe_recall >= 0.95, "{name}: {}", out.after.probe_recall);
    }
}

#[test]
fn isolated_atf_is_repaired_and_found() {
    // ATFs with no Data/Access/Events links: fastmemory's Louvain gets zero edges for them.
    let input = "## [ID: Quarterly_Close]\n**Action:** Close_Books\n**Logic:** Reconcile ledgers before the quarterly close deadline.\n\n## [ID: Vendor_Onboarding]\n**Action:** Onboard_Vendor\n**Logic:** Collect tax forms and banking details from each new vendor.\n";
    let mut m = Memory::new();
    m.ingest(input, Format::Auto, "fin");
    let atfs = m.atfs().to_vec();
    assert!(fastmemory_edges(&atfs).is_empty());
    let p = DensityPolicy::default();
    let before = density::measure(&m, &p);
    assert_eq!(before.isolated_functions, 2);
    assert!(!before.passes);
    let out = density::ensure(&mut m, &p);
    assert!(out.after.passes, "{:#?}", out.after);
    assert!(out.concepts_added > 0);
    let r = query::query(m.graph(), m.index(), "ledgers", 3);
    assert_eq!(r.hits[0].id, "F_Quarterly_Close");
}

#[test]
fn empty_memory_is_an_answer_not_an_error() {
    let m = Memory::new();
    let r = query::query(m.graph(), m.index(), "anything", 5);
    assert_eq!(r.stage, Stage::EmptyMemory);
    assert!(!r.matched && r.handoff && r.hits.is_empty());
    let t = query::traverse(m.graph(), m.index(), "anything", 2, 10);
    assert!(t.nodes.is_empty());
    let rep = density::measure(&m, &DensityPolicy::default());
    assert!(!rep.passes);
}

#[test]
fn cascade_stages_are_reported_honestly() {
    let mut m = Memory::new();
    m.ingest(&fixture("health_science"), Format::Auto, "hs");
    density::ensure(&mut m, &DensityPolicy::default());
    let (g, ix) = (m.graph(), m.index());

    // pick a real vocabulary word and misspell it
    let input = fixture("health_science");
    let word = mahabodi_core::text::terms(&input).into_iter().filter(|t| t.len() >= 9 && t.is_ascii()).max().unwrap();
    let typo: String = word.chars().enumerate().filter(|(i, _)| *i != 3).map(|(_, c)| c).collect();
    let r = query::query(g, ix, &typo, 5);
    assert_eq!(r.stage, Stage::Fuzzy, "{typo}");
    assert!(r.matched);
    assert!(r.corrections.iter().any(|(_, w, _)| *w == word), "{typo} -> {:?}", r.corrections);

    let r = query::query(g, ix, "zzqxv wkkpj", 5);
    assert_eq!(r.stage, Stage::Hub);
    assert!(!r.matched && r.handoff);
    assert!(!r.hits.is_empty() && r.hits.iter().all(|h| h.score == 0.0));
    assert!(r.confidence <= 0.05);
}

#[test]
fn stems_and_cjk_are_searchable() {
    let mut m = Memory::new();
    m.ingest("## [ID: Expense_Rules]\n**Action:** Apply_Rules\n**Logic:** Rules for approving travel expenses.\n", Format::Auto, "x");
    m.ingest("报销需要收据。差旅必须事先批准。", Format::Text, "zh");
    let (g, ix) = (m.graph(), m.index());
    let r = query::query(g, ix, "approve", 3);
    assert!(r.matched && matches!(r.stage, Stage::Stem | Stage::Exact), "{:?}", r.stage);
    assert_eq!(r.hits[0].id, "F_Expense_Rules");
    let r = query::query(g, ix, "收据", 3);
    assert_eq!(r.stage, Stage::Exact);
    assert!(r.hits[0].text.contains("收据"));
}

#[test]
fn traversal_resolves_text_starts_and_stays_bounded() {
    let mut m = Memory::new();
    m.ingest(&fixture("robotics"), Format::Auto, "r");
    density::ensure(&mut m, &DensityPolicy::default());
    let t = query::traverse(m.graph(), m.index(), "ATF_S_1", 2, 15);
    assert_eq!(t.resolved_by, "id");
    assert_eq!(t.start.as_deref(), Some("F_ATF_S_1"));
    assert!(t.nodes.len() <= 15 && t.nodes.len() > 1);
    assert!(t.nodes.iter().all(|n| n.depth <= 2));
    let t = query::traverse(m.graph(), m.index(), "spacecraft", 1, 50);
    assert_eq!(t.resolved_by, "exact");
    let t = query::traverse(m.graph(), m.index(), "qqqzzz", 1, 5);
    assert_eq!(t.resolved_by, "hub");
}

#[test]
fn reingesting_an_id_replaces_it() {
    let mut m = Memory::new();
    m.ingest("## [ID: A]\n**Action:** Old_Thing\n", Format::Auto, "x");
    m.ingest("## [ID: A]\n**Action:** New_Thing\n", Format::Auto, "x");
    assert_eq!(m.atfs().len(), 1);
    assert_eq!(m.atfs()[0].action, "New_Thing");
}

fn token_memory() -> Memory {
    let mut m = Memory::new();
    m.ingest("## [ID: Validate_Token]\n**Action:** Validate_Token\n**Data_Connections:** Session_UUID\n**Logic:** Check the customers session token.\n", Format::Auto, "x");
    m
}

#[test]
fn stopword_and_tiny_queries_do_not_confidently_match() {
    let m = token_memory();
    for q in ["to", "e", "in", "the to", "  "] {
        let r = query::query(m.graph(), m.index(), q, 5);
        assert!(!r.matched && r.handoff, "{q:?} -> {:?} matched={}", r.stage, r.matched);
        assert_eq!(r.term_coverage, 0.0);
    }
}

#[test]
fn one_term_of_many_hands_off() {
    let m = token_memory();
    let r = query::query(m.graph(), m.index(), "refund policy for Khmer wholesale customers", 5);
    assert!(r.matched, "customers is in memory");
    assert!(r.term_coverage < 0.5);
    assert!(r.handoff, "coverage {} must hand off", r.term_coverage);
    let r = query::query(m.graph(), m.index(), "session token", 5);
    assert!(r.matched && !r.handoff);
}

#[test]
fn context_is_empty_on_handoff() {
    let m = token_memory();
    let (c, r) = query::context(m.graph(), m.index(), "zzqxv wkkpj", 5, 2000);
    assert_eq!(r.stage, Stage::Hub);
    assert!(c.is_empty(), "hub fallback leaked into grounding: {c}");
    let (c, _) = query::context(m.graph(), m.index(), "session token", 5, 2000);
    assert!(c.contains("Validate_Token"));
}

#[test]
fn hits_are_memories_never_bare_labels() {
    let mut m = Memory::new();
    m.ingest(&fixture("robotics"), Format::Auto, "r");
    density::ensure(&mut m, &DensityPolicy::default());
    for q in ["mission", "students", "validating spacecraft", "mars"] {
        let r = query::query(m.graph(), m.index(), q, 5);
        assert!(r.hits.iter().all(|h| h.id.starts_with("F_")), "{q}: {:?}", r.hits.iter().map(|h| &h.id).collect::<Vec<_>>());
    }
}

#[test]
fn reingesting_same_prose_does_not_duplicate() {
    let mut m = Memory::new();
    m.ingest("Photosynthesis converts light. Chlorophyll absorbs red light.", Format::Text, "bio");
    m.ingest("Photosynthesis converts light. Chlorophyll absorbs red light.", Format::Text, "bio");
    assert_eq!(m.atfs().len(), 1);
}

#[test]
fn facade_dispatcher_round_trip() {
    use mahabodi_core::Bodi;
    use serde_json::json;
    let b = Bodi::from_json("").unwrap();
    let r = b.call("ingest", &json!({"text": fixture("robotics"), "source": "robotics"})).unwrap();
    assert_eq!(r["density"]["passes"], true, "{r}");
    let q = b.call("query", &json!({"q": "spacecraft", "k": 3})).unwrap();
    assert_eq!(q["matched"], true);
    let snap = b.call("snapshot", &json!({})).unwrap();
    let b2 = Bodi::from_json("{}").unwrap();
    b2.call("restore", &json!({"snapshot": snap})).unwrap();
    assert_eq!(b2.call("query", &json!({"q": "spacecraft", "k": 3})).unwrap()["hits"], q["hits"]);
    assert!(b.call("nope", &json!({})).is_err());
    assert!(b.call("query", &json!({})).is_err());
    assert!(b.call("decide", &json!({"state": "x", "questions": {}})).is_err(), "no model loaded -> error, not panic");
}

#[test]
fn concurrent_reads_and_writes_are_safe() {
    use mahabodi_core::Bodi;
    use serde_json::json;
    let b = std::sync::Arc::new(Bodi::from_json("").unwrap());
    b.call("ingest", &json!({"text": fixture("world_events")})).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let b = b.clone();
            std::thread::spawn(move || {
                for j in 0..20 {
                    if i % 4 == 0 && j % 5 == 0 {
                        b.call("ingest", &json!({"text": format!("Thread {i} note {j} about elevators and space cables."), "format": "text"})).unwrap();
                    } else {
                        let r = b.call("query", &json!({"q": "elevator", "k": 3})).unwrap();
                        assert!(r["hits"].as_array().is_some());
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert!(b.call("stats", &json!({})).unwrap()["atfs"].as_u64().unwrap() > 20);
}

#[test]
fn topology_is_reproducible_and_matches_fastmemory_quality() {
    use mahabodi_core::graph::ClusterEngine;
    for name in FIXTURES {
        let build = |engine| {
            let mut m = Memory::with_engine(engine);
            m.ingest(&fixture(name), Format::Auto, name);
            density::ensure(&mut m, &DensityPolicy::default());
            m
        };
        let a = build(ClusterEngine::Deterministic);
        let b = build(ClusterEngine::Deterministic);
        let blocks = |m: &Memory| m.graph().blocks.iter().map(|b| (b.name.clone(), b.members.iter().map(|&i| m.graph().nodes[i].id.clone()).collect::<Vec<_>>())).collect::<Vec<_>>();
        assert_eq!(blocks(&a), blocks(&b), "{name}: deterministic engine must reproduce communities");

        // quality parity with fastmemory's own (nondeterministic) run: compare modularity
        let q = |m: &Memory| {
            let g = m.graph();
            let mut edges = Vec::new();
            for (i, ns) in g.adj.iter().enumerate() {
                for &j in ns {
                    if i < j {
                        edges.push((g.nodes[i].id.clone(), g.nodes[j].id.clone()));
                    }
                }
            }
            let names: Vec<String> = g.nodes.iter().map(|n| n.id.clone()).collect();
            let comm: Vec<usize> = g.nodes.iter().map(|n| n.block).collect();
            mahabodi_core::louvain::modularity(&edges, &names, &comm)
        };
        let ours = q(&a);
        let theirs: f64 = (0..5).map(|_| q(&build(ClusterEngine::FastMemory))).sum::<f64>() / 5.0;
        assert!(ours >= theirs - 0.02, "{name}: modularity {ours:.3} vs fastmemory mean {theirs:.3}");
    }
}

#[test]
fn ingest_batch_equals_sequential_ingest() {
    use mahabodi_core::Bodi;
    use serde_json::json;
    let docs: Vec<String> = (0..30).map(|i| format!("Passage {i} about elevator cables number {i} and orbital tethers.")).collect();
    let a = Bodi::from_json("").unwrap();
    for (i, d) in docs.iter().enumerate() {
        a.call("ingest", &json!({"text": d, "format": "text", "source": format!("p{i}")})).unwrap();
    }
    let b = Bodi::from_json("").unwrap();
    let batch: Vec<_> = docs.iter().enumerate().map(|(i, d)| json!({"text": d, "format": "text", "source": format!("p{i}")})).collect();
    b.call("ingest_batch", &json!({"docs": batch})).unwrap();
    assert_eq!(a.call("stats", &json!({})).unwrap()["atfs"], b.call("stats", &json!({})).unwrap()["atfs"]);
    for q in ["orbital tethers", "passage 7", "elevator"] {
        let ha: Vec<_> = a.call("query", &json!({"q": q})).unwrap()["hits"].as_array().unwrap().iter().map(|h| h["id"].clone()).collect();
        let hb: Vec<_> = b.call("query", &json!({"q": q})).unwrap()["hits"].as_array().unwrap().iter().map(|h| h["id"].clone()).collect();
        assert_eq!(ha.len(), hb.len(), "{q}");
    }
}

#[test]
fn ingest_batch_equals_sequential_even_with_shared_ids() {
    // fastmemory's fixtures all reuse ids ATF_S_0.. : sequential ingest keeps the newest version of each
    use mahabodi_core::Bodi;
    use serde_json::json;
    let a = Bodi::from_json("").unwrap();
    for name in FIXTURES {
        a.call("ingest", &json!({"text": fixture(name), "source": name})).unwrap();
    }
    let b = Bodi::from_json("").unwrap();
    let docs: Vec<_> = FIXTURES.iter().map(|n| json!({"text": fixture(n), "source": n})).collect();
    b.call("ingest_batch", &json!({"docs": docs})).unwrap();
    let (sa, sb) = (a.call("snapshot", &json!({})).unwrap(), b.call("snapshot", &json!({})).unwrap());
    for k in ["atfs", "texts", "links"] {
        assert_eq!(sa[k], sb[k], "{k} differs between sequential and batch ingest");
    }
    let ids: std::collections::HashSet<_> = sb["atfs"].as_array().unwrap().iter().map(|x| x["id"].clone()).collect();
    assert_eq!(ids.len(), sb["atfs"].as_array().unwrap().len(), "duplicate ATF ids after batch ingest");
    assert_eq!(a.call("stats", &json!({})).unwrap()["nodes"], b.call("stats", &json!({})).unwrap()["nodes"]);
}
