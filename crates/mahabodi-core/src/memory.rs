//! The memory store: ingested ATFs + density concepts, and the graph/index built from them.

use std::collections::HashMap;

use crate::fastmemory::parser::Atf;
use serde::Serialize;

use crate::graph::{ClusterEngine, Graph, GraphInput};
use crate::index::Index;
use crate::ingest::{self, Format};

/// Anything that turns texts into unit vectors (the ONNX embedder implements this).
pub trait TextEmbedder: Send + Sync {
    fn embed_texts(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>>;
}

#[derive(Default)]
pub struct Memory {
    pub(crate) atfs: Vec<Atf>,
    pub(crate) links: Vec<(String, String)>,
    pub(crate) texts: HashMap<String, String>,
    /// (ATF id, concept) edges added by the density guard.
    pub(crate) concepts: Vec<(String, String)>,
    next_passage: usize,
    pub(crate) engine: ClusterEngine,
    graph: Graph,
    index: Index,
    embedder: Option<std::sync::Arc<dyn TextEmbedder>>,
    /// ATF id -> (content hash, unit vector): only new/changed passages are re-embedded.
    dense_cache: HashMap<String, (u64, Vec<f32>)>,
    /// (Function node index, vector) aligned to the current graph.
    dense: Vec<(usize, Vec<f32>)>,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory").field("atfs", &self.atfs.len()).field("dense", &self.dense.len()).finish()
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

#[derive(Debug, Clone, Serialize)]
pub struct IngestReport {
    pub atfs_added: usize,
    pub atfs_total: usize,
    pub nodes: usize,
    pub edges: usize,
    pub blocks: usize,
}

impl Memory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, input: &str, format: Format, source: &str) -> IngestReport {
        self.ingest_many(&[(input, format, source)])
    }

    /// Ingest several documents with ONE rebuild (graph, index, dense vectors). Loading N
    /// documents through N `ingest` calls rebuilds N times (O(N^2) overall); use this for bulk loads.
    pub fn ingest_many(&mut self, docs: &[(&str, Format, &str)]) -> IngestReport {
        // Documents are parsed in parallel (each parse is independent; `next_passage` is only a
        // counter, since passage ids are content-addressed). The result is then merged in document order
        // with the same newest-wins rule as sequential `ingest`: an id re-ingested by a later
        // document (of this batch or a later call) replaces the earlier one, together with the
        // earlier one's links and density concepts. One linear pass replaces the old
        // per-document `retain` over the whole store (O(docs x atfs)). A single rebuild follows.
        use rayon::prelude::*;
        let timing = std::env::var_os("MAHABODI_TIMING").is_some();
        let t0 = std::time::Instant::now();
        let parsed: Vec<(ingest::Ingested, usize)> = docs
            .par_iter()
            .map(|(input, format, source)| {
                let mut n = 0usize;
                let got = ingest::ingest(input, *format, source, &mut n);
                (got, n)
            })
            .collect();
        let added: usize = parsed.iter().map(|(g, _)| g.atfs.len()).sum();
        self.next_passage += parsed.iter().map(|(_, n)| n).sum::<usize>();
        // last batch document that defines each id
        let mut last: HashMap<String, usize> = HashMap::new();
        for (d, (g, _)) in parsed.iter().enumerate() {
            for a in &g.atfs {
                last.insert(a.id.clone(), d);
            }
        }
        self.atfs.retain(|a| !last.contains_key(&a.id));
        self.links.retain(|(a, _)| !last.contains_key(a));
        self.concepts.retain(|(a, _)| !last.contains_key(a));
        for (d, (g, _)) in parsed.into_iter().enumerate() {
            // sequential ingest drops an earlier document's ATF, and its links, when a later one re-defines the id
            let later = |id: &str| last.get(id).is_some_and(|&l| l > d);
            self.atfs.extend(g.atfs.into_iter().filter(|a| !later(&a.id)));
            self.links.extend(g.links.into_iter().filter(|(a, _)| !later(a)));
            self.texts.extend(g.texts);
        }
        if timing {
            eprintln!("[mahabodi] ingest_many parse+merge {:?} ({} docs)", t0.elapsed(), docs.len());
        }
        self.rebuild();
        IngestReport {
            atfs_added: added,
            atfs_total: self.atfs.len(),
            nodes: self.graph.nodes.len(),
            edges: self.graph.edge_count,
            blocks: self.graph.blocks.len(),
        }
    }

    pub fn rebuild(&mut self) {
        let timing = std::env::var_os("MAHABODI_TIMING").is_some();
        let t0 = std::time::Instant::now();
        self.graph = Graph::build(GraphInput {
            atfs: &self.atfs,
            links: &self.links,
            concepts: &self.concepts,
            texts: &self.texts,
            engine: self.engine,
        });
        let t1 = std::time::Instant::now();
        self.index = Index::build(&self.graph);
        let t2 = std::time::Instant::now();
        self.refresh_dense();
        if timing {
            eprintln!("[mahabodi] rebuild graph {:?} index {:?} dense {:?} ({} atfs)", t1 - t0, t2 - t1, t2.elapsed(), self.atfs.len());
        }
    }

    /// Attach (or detach) a dense embedder; existing passages are embedded now.
    pub fn set_embedder(&mut self, e: Option<std::sync::Arc<dyn TextEmbedder>>) {
        self.embedder = e;
        self.dense_cache.clear();
        self.refresh_dense();
    }

    fn refresh_dense(&mut self) {
        self.dense.clear();
        let Some(e) = self.embedder.clone() else { return };
        let g = &self.graph;
        let items: Vec<(usize, String, String)> = g
            .functions()
            .map(|i| (i, g.nodes[i].id.clone(), format!("{}\n{}", g.nodes[i].label.replace('_', " "), g.nodes[i].text)))
            .collect();
        let missing: Vec<&(usize, String, String)> =
            items.iter().filter(|(_, id, t)| self.dense_cache.get(id).map_or(true, |(h, _)| *h != fnv(t))).collect();
        if !missing.is_empty() {
            let texts: Vec<String> = missing.iter().map(|x| x.2.clone()).collect();
            // a failed embed leaves hybrid retrieval off rather than failing the ingest
            if let Ok(vs) = e.embed_texts(&texts) {
                for ((_, id, t), v) in missing.iter().zip(vs) {
                    self.dense_cache.insert(id.clone(), (fnv(t), v));
                }
            }
        }
        self.dense = items.iter().filter_map(|(i, id, _)| self.dense_cache.get(id).map(|(_, v)| (*i, v.clone()))).collect();
    }

    pub fn dense(&self) -> &[(usize, Vec<f32>)] {
        &self.dense
    }

    pub fn embedder(&self) -> Option<std::sync::Arc<dyn TextEmbedder>> {
        self.embedder.clone()
    }

    /// fastmemory's own topology JSON for this memory (computed on demand with
    /// `fastmemory::cluster::run_louvain_inline`), for `fastmemory_search` comparisons.
    pub fn fastmemory_json(&self) -> String {
        let input = GraphInput { atfs: &self.atfs, links: &self.links, concepts: &self.concepts, texts: &self.texts, engine: ClusterEngine::FastMemory };
        crate::fastmemory::cluster::run_louvain_inline(&crate::graph::all_edges(&input), &self.atfs)
    }

    pub fn with_engine(engine: ClusterEngine) -> Self {
        Memory { engine, ..Default::default() }
    }

    pub fn clear(&mut self) {
        let e = self.embedder.take();
        *self = Memory::with_engine(self.engine);
        self.embedder = e;
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    pub fn atfs(&self) -> &[Atf] {
        &self.atfs
    }

    pub fn text_of(&self, atf_id: &str) -> Option<&str> {
        self.texts.get(atf_id).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-parallel `ingest_many`: parse and replace one document at a time.
    fn sequential(m: &mut Memory, docs: &[(&str, Format, &str)]) {
        for (input, format, source) in docs {
            let got = ingest::ingest(input, *format, source, &mut m.next_passage);
            let new_ids: std::collections::HashSet<&str> = got.atfs.iter().map(|a| a.id.as_str()).collect();
            m.atfs.retain(|a| !new_ids.contains(a.id.as_str()));
            m.links.retain(|(a, _)| !new_ids.contains(a.as_str()));
            m.concepts.retain(|(a, _)| !new_ids.contains(a.as_str()));
            m.atfs.extend(got.atfs);
            m.links.extend(got.links);
            m.texts.extend(got.texts);
        }
        m.rebuild();
    }

    fn state(m: &Memory) -> String {
        let mut texts: Vec<_> = m.texts.iter().collect();
        texts.sort();
        format!("{:?}|{:?}|{:?}|{:?}|{}", m.atfs, m.links, m.concepts, texts, m.next_passage)
    }

    #[test]
    fn parallel_ingest_matches_sequential_replacement() {
        // overlapping ATF ids across documents and batches, links, prose and entity tags
        let atf = |id: &str, act: &str, link: &str| format!("## [ID: {id}]\n**Action:** {act}\n**Input:** {{Order_Id}}\n**Context_Links:** [{link}]\n");
        let b1: Vec<String> = vec![
            atf("A", "First_A", "B"),
            atf("B", "First_B", "A") + &atf("C", "First_C", "A"),
            "Refunds take five days. Escalate after two.\n\nInvoices are sent monthly.".into(),
            atf("A", "Second_A", "C"), // re-defines A inside the same batch
            "(Component Billing) (Function Charge) uses (Data Card_Token).".into(),
        ];
        let b2: Vec<String> = vec![atf("C", "Newer_C", "B"), "Refunds take five days. Escalate after two.".into(), atf("D", "Only_D", "A")];
        fn docs(v: &[String]) -> Vec<(&str, Format, &str)> {
            v.iter().enumerate().map(|(i, s)| (s.as_str(), Format::Auto, if i % 2 == 0 { "kb" } else { "notes" })).collect()
        }
        let (mut par, mut seq) = (Memory::new(), Memory::new());
        par.concepts.push(("C".into(), "Concept_x".into())); // a density concept on an id that gets replaced
        seq.concepts.push(("C".into(), "Concept_x".into()));
        for b in [&b1, &b2] {
            par.ingest_many(&docs(b));
            sequential(&mut seq, &docs(b));
            assert_eq!(state(&par), state(&seq));
        }
        assert!(par.atfs.iter().any(|a| a.id == "A" && a.action == "Second_A"));
        assert_eq!(par.atfs.iter().filter(|a| a.id == "A").count(), 1);
        assert!(!par.concepts.iter().any(|(a, _)| a == "C"));
    }
}
