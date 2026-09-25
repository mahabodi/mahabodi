//! The memory topology: Louvain communities over ATF edges.
//!
//! Edges are built exactly as fastmemory's CLI builds them (`F_x -> D_/A_/E_`),
//! plus Bodi's own `F_a -- F_b` context links and `F_x -- K_concept` density
//! edges. Communities come from fastmemory's inline Louvain: by default the
//! deterministic port in `louvain.rs` (identical input -> identical communities),
//! or fastmemory's own `cluster::run_louvain_inline` with `ClusterEngine::FastMemory`.
//! Bodi keeps every ATF as a node even when it has no edges (fastmemory drops
//! those, which is one way its queries silently miss).

use std::collections::{HashMap, HashSet};

use crate::fastmemory::parser::Atf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Function,
    Data,
    Access,
    Event,
    Concept,
}

impl Level {
    pub fn from_id(id: &str) -> Level {
        match id.get(..2) {
            Some("F_") => Level::Function,
            Some("D_") => Level::Data,
            Some("A_") => Level::Access,
            Some("E_") => Level::Event,
            _ => Level::Concept,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub id: String,
    pub level: Level,
    /// Human label: ATF id + action for functions, bare name otherwise.
    pub label: String,
    /// Retrievable passage/logic text (functions only).
    pub text: String,
    pub block: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Block {
    pub name: String,
    pub members: Vec<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub index_of: HashMap<String, usize>,
    pub adj: Vec<Vec<usize>>,
    pub blocks: Vec<Block>,
    pub edge_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterEngine {
    /// `louvain::communities`: fastmemory's algorithm with deterministic tie-breaking.
    #[default]
    Deterministic,
    /// `fastmemory::cluster::run_louvain_inline` itself (HashMap-order ties: may vary per process).
    FastMemory,
}

pub struct GraphInput<'a> {
    pub atfs: &'a [Atf],
    pub links: &'a [(String, String)],
    pub concepts: &'a [(String, String)],
    pub texts: &'a HashMap<String, String>,
    pub engine: ClusterEngine,
}

/// All edges Bodi clusters on: fastmemory's, plus context links and density concepts.
pub fn all_edges(input: &GraphInput<'_>) -> Vec<(String, String)> {
    let mut edges = fastmemory_edges(input.atfs);
    let known: HashSet<&str> = input.atfs.iter().map(|a| a.id.as_str()).collect();
    for (a, b) in input.links {
        if known.contains(b.as_str()) && a != b {
            edges.push((format!("F_{a}"), format!("F_{b}")));
        }
    }
    for (f, k) in input.concepts {
        edges.push((format!("F_{f}"), format!("K_{k}")));
    }
    edges
}

/// fastmemory's edge construction (src/main.rs `resolve_and_build`), verbatim in behaviour.
pub fn fastmemory_edges(atfs: &[Atf]) -> Vec<(String, String)> {
    let mut edges = Vec::new();
    for atf in atfs {
        let f_id = format!("F_{}", atf.id);
        for link in &atf.data_connections {
            edges.push((f_id.clone(), format!("D_{}", link)));
        }
        for acc in atf.access.split(',') {
            let acc = acc.trim();
            if !acc.is_empty() {
                edges.push((f_id.clone(), format!("A_{}", acc)));
            }
        }
        for ev in atf.events.split(',') {
            let ev = ev.trim();
            if !ev.is_empty() {
                edges.push((f_id.clone(), format!("E_{}", ev)));
            }
        }
    }
    edges
}

impl Graph {
    pub fn build(input: GraphInput<'_>) -> Graph {
        let edges = all_edges(&input);
        let groups: Vec<Vec<String>> = match input.engine {
            ClusterEngine::Deterministic => {
                let (names, comm) = crate::louvain::communities(&edges);
                let k = comm.iter().copied().max().map_or(0, |m| m + 1);
                let mut groups = vec![Vec::new(); k];
                for (n, c) in names.into_iter().zip(comm) {
                    groups[c].push(n);
                }
                groups
            }
            ClusterEngine::FastMemory => {
                let json = crate::fastmemory::cluster::run_louvain_inline(&edges, &input.atfs.to_vec());
                let mut groups = Vec::new();
                if let Ok(serde_json::Value::Array(blocks)) = serde_json::from_str::<serde_json::Value>(&json) {
                    for b in blocks {
                        let ids: Vec<String> = b.get("nodes").and_then(|v| v.as_array()).into_iter().flatten()
                            .filter_map(|n| n.get("id").and_then(|v| v.as_str()).map(String::from)).collect();
                        groups.push(ids);
                    }
                }
                groups
            }
        };

        let mut g = Graph::default();
        let actions: HashMap<&str, &Atf> = input.atfs.iter().map(|a| (a.id.as_str(), a)).collect();

        let add = |g: &mut Graph, id: &str| -> usize {
            if let Some(&i) = g.index_of.get(id) {
                return i;
            }
            let level = Level::from_id(id);
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
            let i = g.nodes.len();
            g.nodes.push(Node { id: id.to_string(), level, label, text, block: usize::MAX });
            g.index_of.insert(id.to_string(), i);
            g.adj.push(Vec::new());
            i
        };

        // every ATF is a node, connected or not
        for a in input.atfs {
            add(&mut g, &format!("F_{}", a.id));
        }
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for (s, t) in &edges {
            let si = add(&mut g, s);
            let ti = add(&mut g, t);
            let key = (si.min(ti), si.max(ti));
            if si != ti && seen.insert(key) {
                g.adj[si].push(ti);
                g.adj[ti].push(si);
            }
        }
        g.edge_count = seen.len();

        // community membership; block names follow fastmemory's scheme
        for (ci, ids) in groups.iter().enumerate() {
            let members: Vec<usize> = ids.iter().filter_map(|id| g.index_of.get(id).copied()).collect();
            if members.is_empty() {
                continue;
            }
            let funcs: Vec<&str> = members.iter().filter(|&&m| g.nodes[m].level == Level::Function).map(|&m| &g.nodes[m].id[2..]).collect();
            let name = match funcs.len() {
                1 => format!("C - {}", funcs[0]),
                0 => format!("C - Data_{ci}"),
                _ => format!("C - Community_{ci}"),
            };
            let bi = g.blocks.len();
            for &m in &members {
                g.nodes[m].block = bi;
            }
            g.blocks.push(Block { name, members });
        }
        // isolated nodes (no edges => absent from fastmemory output) get singleton blocks
        for i in 0..g.nodes.len() {
            if g.nodes[i].block == usize::MAX {
                let bi = g.blocks.len();
                g.nodes[i].block = bi;
                g.blocks.push(Block { name: format!("C - {}", &g.nodes[i].id[2..]), members: vec![i] });
            }
        }
        // deterministic block order (fastmemory's own engine iterates a HashMap)
        let mut order: Vec<usize> = (0..g.blocks.len()).collect();
        order.sort_by(|&a, &b| {
            let ka = g.blocks[a].members.iter().map(|&m| g.nodes[m].id.as_str()).min();
            let kb = g.blocks[b].members.iter().map(|&m| g.nodes[m].id.as_str()).min();
            ka.cmp(&kb)
        });
        let mut remap = vec![0; order.len()];
        let mut blocks = Vec::with_capacity(order.len());
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new;
            let mut b = g.blocks[old].clone();
            b.members.sort_unstable();
            blocks.push(b);
        }
        g.blocks = blocks;
        for n in &mut g.nodes {
            n.block = remap[n.block];
        }
        g
    }

    pub fn degree(&self, i: usize) -> usize {
        self.adj[i].len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn functions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.nodes.len()).filter(|&i| self.nodes[i].level == Level::Function)
    }
}
