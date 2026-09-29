//! Pure derivation of what the PostgreSQL store persists, using exactly the rules of the in-process
//! `Graph::build` + `Index::build`: node ids/levels/labels/texts, links, and per-node weighted term and stem
//! frequencies (label terms weight 3.0, text terms 1.0). No database here; `store::pg` writes these rows.

use std::collections::{BTreeMap, HashMap};

use crate::fastmemory::parser::Atf;
use crate::graph::{all_edges, GraphInput, Level};
use crate::text;

/// One graph node as the store keeps it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeRow {
    pub id: String,
    pub level: Level,
    pub label: String,
    pub text: String,
    /// Weighted term frequency, keyed and ordered by term (label terms x3.0, text terms x1.0).
    pub tf: BTreeMap<String, f32>,
    /// The same weights summed per `text::stem` of each term.
    pub stf: BTreeMap<String, f32>,
    /// Sum of `tf` weights: BM25 document length.
    pub len: f32,
}

const W_LABEL: f32 = 3.0; // index.rs
const W_TEXT: f32 = 1.0;

/// Nodes and undirected links for a set of ATFs, as `Graph::build` would create them (without communities).
/// Node order: every ATF's Function node first (ATF order), then other nodes in first-edge order.
pub fn derive(input: &GraphInput<'_>) -> (Vec<NodeRow>, Vec<(String, String)>) {
    let actions: HashMap<&str, &Atf> = input.atfs.iter().map(|a| (a.id.as_str(), a)).collect();
    let mut order: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for a in input.atfs {
        let id = format!("F_{}", a.id);
        if seen.insert(id.clone()) {
            order.push(id);
        }
    }
    let edges = all_edges(input);
    let mut links: Vec<(String, String)> = Vec::new();
    let mut seen_edge: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for (s, t) in &edges {
        for n in [s, t] {
            if seen.insert(n.clone()) {
                order.push(n.clone());
            }
        }
        if s != t {
            let key = if s < t { (s.clone(), t.clone()) } else { (t.clone(), s.clone()) };
            if seen_edge.insert(key.clone()) {
                links.push(key);
            }
        }
    }
    let nodes = order
        .into_iter()
        .map(|id| {
            let level = Level::from_id(&id);
            let bare = &id[2..];
            let (label, text) = match level {
                Level::Function => {
                    let a = actions.get(bare);
                    let action = a.map(|a| a.action.as_str()).unwrap_or("");
                    let label = if action.is_empty() || action == bare { bare.to_string() } else { format!("{bare} {action}") };
                    let text = input.texts.get(bare).cloned().unwrap_or_else(|| {
                        a.map(|a| format!("{} {}", a.input, a.logic).trim().to_string()).unwrap_or_default()
                    });
                    (label, text)
                }
                _ => (bare.to_string(), String::new()),
            };
            let mut tf: BTreeMap<String, f32> = BTreeMap::new();
            for t in text::terms(&label) {
                *tf.entry(t).or_default() += W_LABEL;
            }
            for t in text::terms(&text) {
                *tf.entry(t).or_default() += W_TEXT;
            }
            let mut stf: BTreeMap<String, f32> = BTreeMap::new();
            for (t, w) in &tf {
                *stf.entry(text::stem(t)).or_default() += *w;
            }
            let len = tf.values().sum();
            NodeRow { id, level, label, text, tf, stf, len }
        })
        .collect();
    (nodes, links)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ClusterEngine, Graph};
    use crate::ingest::{ingest, Format};

    fn fixture() -> (Vec<Atf>, Vec<(String, String)>, HashMap<String, String>) {
        let docs = [
            ("# Barack Obama\n\nBarack Hussein Obama II is an American politician who served as the 44th president.", "pg1"),
            ("# Refund policy\n\nRefunds take five days. Escalate after two days to the billing team.", "kb"),
            ("## [ID: ATF_REF_01]\n**Action:** Approve_Refund\n**Input:** {Order_Id}\n**Context_Links:** [ATF_REF_02]\n## [ID: ATF_REF_02]\n**Action:** Notify_Customer\n**Input:** {Email}", "atf"),
            ("UUIDToken rotation uses B2B keys; 東京 offices follow the same policy.", "misc"),
        ];
        let (mut atfs, mut links, mut texts) = (Vec::new(), Vec::new(), HashMap::new());
        for (d, s) in docs {
            let mut n = 0;
            let g = ingest(d, Format::Auto, s, &mut n);
            atfs.extend(g.atfs);
            links.extend(g.links);
            texts.extend(g.texts);
        }
        (atfs, links, texts)
    }

    /// The store's rows must be exactly the in-process graph's nodes (ids, levels, labels, texts) and edges.
    #[test]
    fn derive_matches_graph_build() {
        let (atfs, links, texts) = fixture();
        let concepts = vec![("pg_1_x".to_string(), "Politics".to_string())];
        let input = GraphInput { atfs: &atfs, links: &links, concepts: &concepts, texts: &texts, engine: ClusterEngine::Deterministic };
        let (nodes, edges) = derive(&input);
        let g = Graph::build(GraphInput { atfs: &atfs, links: &links, concepts: &concepts, texts: &texts, engine: ClusterEngine::Deterministic });
        assert_eq!(nodes.len(), g.nodes.len());
        for (a, b) in nodes.iter().zip(&g.nodes) {
            assert_eq!((&a.id, a.level, &a.label, &a.text), (&b.id, b.level, &b.label, &b.text));
        }
        // weighted term/stem frequencies and lengths equal the in-process index's postings
        let ix = crate::index::Index::build(&g);
        let (exact, stems, doc_len) = ix.parity_view();
        for (i, n) in nodes.iter().enumerate() {
            assert!((n.len - doc_len[i]).abs() < 1e-5, "len {}", n.id);
            for (t, w) in &n.tf {
                let p = exact[t].iter().find(|(d, _)| *d == i).expect("posting");
                assert!((p.1 - w).abs() < 1e-5, "tf {} {}", n.id, t);
            }
            for (t, w) in &n.stf {
                let p = stems[t].iter().find(|(d, _)| *d == i).expect("stem posting");
                assert!((p.1 - w).abs() < 1e-5, "stf {} {}", n.id, t);
            }
        }
        let total: usize = exact.values().map(|v| v.len()).sum();
        assert_eq!(total, nodes.iter().map(|n| n.tf.len()).sum::<usize>());
        assert_eq!(edges.len(), g.edge_count);
        for (s, t) in &edges {
            let (si, ti) = (g.index_of[s], g.index_of[t]);
            assert!(g.adj[si].contains(&ti));
        }
    }
}
