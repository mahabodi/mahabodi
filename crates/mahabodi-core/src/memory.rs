//! The memory store: ingested ATFs + density concepts, and the graph/index built from them.

use std::collections::HashMap;

use fastmemory::parser::Atf;
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
        // Exactly the per-document steps of sequential `ingest` (re-ingesting an id replaces it,
        // newest wins - also between documents of one batch), with a single rebuild at the end.
        let mut added = 0;
        for (input, format, source) in docs {
            let got = ingest::ingest(input, *format, source, &mut self.next_passage);
            added += got.atfs.len();
            let new_ids: std::collections::HashSet<&str> = got.atfs.iter().map(|a| a.id.as_str()).collect();
            self.atfs.retain(|a| !new_ids.contains(a.id.as_str()));
            self.links.retain(|(a, _)| !new_ids.contains(a.as_str()));
            self.concepts.retain(|(a, _)| !new_ids.contains(a.as_str()));
            self.atfs.extend(got.atfs);
            self.links.extend(got.links);
            self.texts.extend(got.texts);
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
        self.graph = Graph::build(GraphInput {
            atfs: &self.atfs,
            links: &self.links,
            concepts: &self.concepts,
            texts: &self.texts,
            engine: self.engine,
        });
        self.index = Index::build(&self.graph);
        self.refresh_dense();
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
        fastmemory::cluster::run_louvain_inline(&crate::graph::all_edges(&input), &self.atfs)
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
