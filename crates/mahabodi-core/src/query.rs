//! Query cascade, traversal and context assembly.
//!
//! "Never fails" is made precise here: for any query on a non-empty memory the
//! result has at least one hit, and `stage` says how it was found:
//!
//!   exact -> substring (fastmemory's own rule) -> stem -> fuzzy (trigram) -> hub
//!
//! `hub` is the explicit low-confidence fallback (most connected memories); it is
//! reported as such, never passed off as a match. An empty memory returns
//! `stage = empty_memory` with no hits: an honest answer, not an error.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::Serialize;
use serde_json::Value;

use crate::graph::{Graph, Level};
use crate::index::{level_rank, Index};
use crate::text;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Exact,
    Substring,
    Stem,
    Fuzzy,
    /// No lexical match, but a passage is semantically close (dense similarity >= threshold).
    Dense,
    Hub,
    EmptyMemory,
}

impl Stage {
    fn base_confidence(self) -> f64 {
        match self {
            Stage::Exact => 1.0,
            Stage::Substring => 0.8,
            Stage::Stem => 0.75,
            Stage::Fuzzy => 0.5,
            Stage::Dense => 0.5,
            Stage::Hub => 0.05,
            Stage::EmptyMemory => 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub id: String,
    pub label: String,
    pub level: Level,
    pub block: String,
    pub score: f64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BlockContext {
    pub name: String,
    /// Sibling nodes of the block (fastmemory's "deepest encompassing block"), capped.
    pub members: Vec<String>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub query: String,
    /// False for `hub` and `empty_memory`: nothing in memory matched the query.
    pub matched: bool,
    /// True when System 1 should not act on this result and should hand off to System 2
    /// (no match, or only a weak fuzzy match covering under half the query terms).
    pub handoff: bool,
    pub stage: Stage,
    pub confidence: f64,
    /// Fraction of the query's content terms that matched something (1.0 for substring/hub-free exact).
    pub term_coverage: f64,
    pub hits: Vec<Hit>,
    pub blocks: Vec<BlockContext>,
    /// Cosine similarity of the best dense match (hybrid retrieval only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_similarity: Option<f64>,
    /// Query term -> vocabulary term substitutions made by the fuzzy stage.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<(String, String, f64)>,
}

const BLOCK_MEMBER_CAP: usize = 24;

/// Dense side of hybrid retrieval: unit vectors of Function nodes and the query's vector.
pub struct Dense<'a> {
    pub vecs: &'a [(usize, Vec<f32>)],
    pub query: Vec<f32>,
    /// A dense match at or above this cosine counts as a match on its own and waives the
    /// term-coverage handoff.
    pub min_similarity: f32,
    /// Opt-in caution: a fuzzy (typo-corrected) match hands off unless its top answer has
    /// dense similarity >= min_similarity. Off by default: on SQuAD train it cut confidently-wrong
    /// answers but cut correct answers more (research/results/tune_hybrid_fuzzycheck.json).
    pub fuzzy_requires_dense: bool,
}

pub fn query(g: &Graph, ix: &Index, q: &str, k: usize) -> QueryResult {
    query_with(g, ix, q, k, None)
}

/// Lexical cascade, optionally fused with dense retrieval by Reciprocal Rank Fusion (k=60).
pub fn query_with(g: &Graph, ix: &Index, q: &str, k: usize, dense: Option<&Dense<'_>>) -> QueryResult {
    let k = k.max(1);
    let mut res = QueryResult {
        query: q.to_string(),
        matched: false,
        handoff: true,
        stage: Stage::EmptyMemory,
        confidence: 0.0,
        term_coverage: 0.0,
        hits: Vec::new(),
        blocks: Vec::new(),
        dense_similarity: None,
        corrections: Vec::new(),
    };
    if g.is_empty() {
        return res;
    }
    let terms = text::terms(q);
    // A query with no content terms (stopwords, 1-2 chars) has nothing to cover: 0.0, not 1.0.
    let covered = |scores_terms: &dyn Fn(&str) -> bool| -> f64 {
        if terms.is_empty() {
            0.0
        } else {
            terms.iter().filter(|t| scores_terms(t)).count() as f64 / terms.len() as f64
        }
    };

    let mut scores = ix.search_exact(&terms);
    // fastmemory's substring rule only for queries that carry content: "to" or "e" would
    // otherwise "match" every id containing those letters.
    let substring = if terms.is_empty() || q.trim().chars().count() < 3 { Vec::new() } else { substring_hits(g, q) };
    let (mut stage, coverage) = if !scores.is_empty() {
        boost(&mut scores, &substring);
        (Stage::Exact, covered(&|t| ix.has_term(t)))
    } else if !substring.is_empty() {
        scores = substring.iter().map(|&i| (i, 1.0f32)).collect();
        (Stage::Substring, covered(&|t| substring.iter().any(|&i| {
            let n = &g.nodes[i];
            n.id.to_lowercase().contains(t) || n.label.to_lowercase().contains(t)
        })))
    } else {
        scores = ix.search_stem(&terms);
        if !scores.is_empty() {
            (Stage::Stem, covered(&|t| !ix.search_stem(&[t.to_string()]).is_empty()))
        } else {
            let (s, used) = ix.search_fuzzy(&terms, 0.4);
            scores = s;
            if !scores.is_empty() {
                let hit_terms: HashSet<&str> = used.iter().map(|(t, _, _)| t.as_str()).collect();
                let cov = covered(&|t| hit_terms.contains(t));
                res.corrections = used;
                (Stage::Fuzzy, cov)
            } else {
                scores = Index::hubs(g, k).into_iter().map(|i| (i, 0.0f32)).collect();
                (Stage::Hub, 0.0)
            }
        }
    };

    // Hits are memories (ATFs): a matched data/concept/access/event node passes its score to
    // the ATFs it links (split by its degree), so "Receipt" finds the passages that mention
    // receipts rather than a bare label, and stemmer-made K_ labels never surface as hits.
    if stage != Stage::Hub {
        let mut spread: HashMap<usize, f32> = HashMap::new();
        for (&i, &s) in &scores {
            if g.nodes[i].level == Level::Function {
                *spread.entry(i).or_default() += s;
            } else {
                let fs: Vec<usize> = g.adj[i].iter().copied().filter(|&j| g.nodes[j].level == Level::Function).collect();
                let share = 0.5 * s / (fs.len() as f32).sqrt().max(1.0);
                for f in fs {
                    *spread.entry(f).or_default() += share;
                }
            }
        }
        if !spread.is_empty() {
            scores = spread;
        }
    }
    let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(level_rank(g.nodes[a.0].level).cmp(&level_rank(g.nodes[b.0].level)))
            .then(g.degree(b.0).cmp(&g.degree(a.0)))
            .then(g.nodes[a.0].id.cmp(&g.nodes[b.0].id))
    });
    let mut dense_ok = false;
    let mut top_hit_sim: Option<f32> = None;
    if let Some(d) = dense.filter(|d| !d.vecs.is_empty()) {
        let mut dr: Vec<(usize, f32)> = d.vecs.iter().map(|(i, v)| (*i, v.iter().zip(&d.query).map(|(a, b)| a * b).sum())).collect();
        dr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        dr.truncate(50);
        let top_sim = dr.first().map(|x| x.1).unwrap_or(0.0);
        res.dense_similarity = Some(top_sim as f64);
        dense_ok = top_sim >= d.min_similarity;
        if stage == Stage::Hub {
            if dense_ok {
                stage = Stage::Dense;
                ranked = dr;
            }
        } else {
            let mut fused: HashMap<usize, f32> = HashMap::new();
            for (r, (i, _)) in ranked.iter().take(50).enumerate() {
                *fused.entry(*i).or_default() += 1.0 / (60.0 + r as f32 + 1.0);
            }
            for (r, (i, _)) in dr.iter().enumerate() {
                *fused.entry(*i).or_default() += 1.0 / (60.0 + r as f32 + 1.0);
            }
            ranked = fused.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(g.nodes[a.0].id.cmp(&g.nodes[b.0].id)));
        }
    }
    ranked.truncate(k);
    if let (Some(d), Some(&(t, _))) = (dense.filter(|d| !d.vecs.is_empty()), ranked.first()) {
        top_hit_sim = d.vecs.iter().find(|(i, _)| *i == t).map(|(_, v)| v.iter().zip(&d.query).map(|(a, b)| a * b).sum());
    }

    let top = ranked.first().map(|r| r.1).unwrap_or(0.0).max(1e-9);
    let mut seen_blocks = Vec::new();
    for (i, s) in &ranked {
        let n = &g.nodes[*i];
        res.hits.push(Hit {
            id: n.id.clone(),
            label: n.label.clone(),
            level: n.level,
            block: g.blocks[n.block].name.clone(),
            score: if stage == Stage::Hub { 0.0 } else { (*s / top) as f64 },
            text: n.text.clone(),
        });
        if !seen_blocks.contains(&n.block) {
            seen_blocks.push(n.block);
        }
    }
    for b in seen_blocks {
        let blk = &g.blocks[b];
        let mut members: Vec<usize> = blk.members.clone();
        members.sort_by_key(|&m| (level_rank(g.nodes[m].level), std::cmp::Reverse(g.degree(m))));
        res.blocks.push(BlockContext {
            name: blk.name.clone(),
            members: members.iter().take(BLOCK_MEMBER_CAP).map(|&m| g.nodes[m].id.clone()).collect(),
            size: blk.members.len(),
        });
    }
    res.stage = stage;
    res.matched = !matches!(stage, Stage::Hub | Stage::EmptyMemory);
    // Matching one term of a many-term query is not an answer to it, whatever the stage.
    // A strong dense match answers a paraphrase even when few query terms appear verbatim.
    res.handoff = !res.matched || (coverage < 0.5 && !dense_ok);
    // A typo-corrected (fuzzy) match is only trusted when the answer it picked is also
    // semantically close to the query; otherwise the correction probably guessed wrong.
    if stage == Stage::Fuzzy {
        if let (Some(d), Some(sim)) = (dense.filter(|d| d.fuzzy_requires_dense), top_hit_sim) {
            if sim < d.min_similarity {
                res.handoff = true;
            }
        }
    }
    res.term_coverage = coverage;
    res.confidence = if stage == Stage::Dense {
        (stage.base_confidence() * res.dense_similarity.unwrap_or(0.0)).min(1.0)
    } else {
        (stage.base_confidence() * (0.5 + 0.5 * coverage.max(if dense_ok { 0.5 } else { 0.0 }))).min(1.0)
    };
    res
}

/// fastmemory's matching rule: case-insensitive substring of the whole query in id/label.
fn substring_hits(g: &Graph, q: &str) -> Vec<usize> {
    let ql = q.trim().to_lowercase();
    if ql.is_empty() {
        return Vec::new();
    }
    (0..g.nodes.len())
        .filter(|&i| g.nodes[i].id.to_lowercase().contains(&ql) || g.nodes[i].label.to_lowercase().contains(&ql))
        .collect()
}

fn boost(scores: &mut HashMap<usize, f32>, substring: &[usize]) {
    let top = scores.values().cloned().fold(0.0f32, f32::max);
    for &i in substring {
        *scores.entry(i).or_default() += 0.25 * top;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TraversalNode {
    pub id: String,
    pub label: String,
    pub level: Level,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Traversal {
    pub start: Option<String>,
    /// How `start` was resolved: `id` (exact node id) or a query stage.
    pub resolved_by: String,
    pub nodes: Vec<TraversalNode>,
    pub edges: Vec<(String, String)>,
}

/// BFS from `start` (a node id, bare ATF id, or any text resolved through the query cascade).
pub fn traverse(g: &Graph, ix: &Index, start: &str, hops: usize, limit: usize) -> Traversal {
    let mut t = Traversal { start: None, resolved_by: "empty_memory".into(), nodes: Vec::new(), edges: Vec::new() };
    if g.is_empty() {
        return t;
    }
    let exact = g
        .index_of
        .get(start)
        .or_else(|| ["F_", "D_", "K_", "A_", "E_"].iter().find_map(|p| g.index_of.get(&format!("{p}{start}"))))
        .copied();
    let s = match exact {
        Some(i) => {
            t.resolved_by = "id".into();
            i
        }
        None => {
            let r = query(g, ix, start, 1);
            t.resolved_by = serde_json::to_value(r.stage).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
            g.index_of[&r.hits[0].id]
        }
    };
    t.start = Some(g.nodes[s].id.clone());
    let limit = limit.max(1);
    let mut depth = vec![usize::MAX; g.nodes.len()];
    let mut q = VecDeque::from([s]);
    depth[s] = 0;
    let mut edge_set = HashSet::new();
    while let Some(u) = q.pop_front() {
        t.nodes.push(TraversalNode { id: g.nodes[u].id.clone(), label: g.nodes[u].label.clone(), level: g.nodes[u].level, depth: depth[u] });
        if t.nodes.len() >= limit {
            break;
        }
        if depth[u] >= hops {
            continue;
        }
        let mut nbrs = g.adj[u].clone();
        nbrs.sort_by_key(|&v| (level_rank(g.nodes[v].level), std::cmp::Reverse(g.degree(v)), v));
        for v in nbrs {
            if depth[v] == usize::MAX {
                depth[v] = depth[u] + 1;
                q.push_back(v);
            }
            let key = (u.min(v), u.max(v));
            if edge_set.insert(key) {
                t.edges.push((g.nodes[key.0].id.clone(), g.nodes[key.1].id.clone()));
            }
        }
    }
    let kept: HashSet<&str> = t.nodes.iter().map(|n| n.id.as_str()).collect();
    t.edges.retain(|(a, b)| kept.contains(a.as_str()) && kept.contains(b.as_str()));
    t
}

/// Grounding text for a System-1 decision: top hits' passages plus their block siblings.
/// Empty when the query hands off: unrelated memory must never be passed as grounding.
pub fn context(g: &Graph, ix: &Index, q: &str, k: usize, max_chars: usize) -> (String, QueryResult) {
    context_with(g, ix, q, k, max_chars, None)
}

pub fn context_with(g: &Graph, ix: &Index, q: &str, k: usize, max_chars: usize, dense: Option<&Dense<'_>>) -> (String, QueryResult) {
    let r = query_with(g, ix, q, k, dense);
    let mut out = String::new();
    if r.handoff {
        return (out, r);
    }
    for h in &r.hits {
        let line = if h.text.is_empty() { format!("- {}\n", h.label) } else { format!("- {}: {}\n", h.label, h.text) };
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);
    }
    for b in &r.blocks {
        let sib: Vec<&str> = b.members.iter().filter_map(|id| g.index_of.get(id)).map(|&i| g.nodes[i].label.as_str()).take(8).collect();
        let line = format!("[{}] related: {}\n", b.name, sib.join(", "));
        if out.len() + line.len() > max_chars {
            break;
        }
        out.push_str(&line);
    }
    (out, r)
}

/// fastmemory's own `query::search_memory` (src/query.rs, MIT), ported verbatim in behaviour so
/// Bodi can report what plain fastmemory would have returned for the same memory and query.
pub fn fastmemory_search(memory_json: &str, query: &str) -> Value {
    let memory: Value = serde_json::from_str(memory_json).unwrap_or(Value::Null);
    let mut results = Vec::new();
    if let Some(blocks) = memory.as_array() {
        for block in blocks {
            if let Some(d) = fm_deepest(block, query) {
                results.push(d);
            }
        }
    }
    Value::Array(results)
}

fn fm_has_match(val: &Value, q: &str) -> bool {
    let q = q.to_lowercase();
    let Some(obj) = val.as_object() else { return false };
    for key in ["name", "action", "id"] {
        if obj.get(key).and_then(|v| v.as_str()).is_some_and(|s| s.to_lowercase().contains(&q)) {
            return true;
        }
    }
    for key in ["nodes", "sub_blocks"] {
        if obj.get(key).and_then(|v| v.as_array()).is_some_and(|a| a.iter().any(|n| fm_has_match(n, &q))) {
            return true;
        }
    }
    false
}

fn fm_deepest(block: &Value, q: &str) -> Option<Value> {
    if !fm_has_match(block, q) {
        return None;
    }
    if let Some(subs) = block.get("sub_blocks").and_then(|v| v.as_array()) {
        let deeper: Vec<Value> = subs.iter().filter_map(|sb| fm_deepest(sb, q)).collect();
        if !deeper.is_empty() {
            let mut c = block.clone();
            c["sub_blocks"] = Value::Array(deeper);
            return Some(c);
        }
    }
    let ql = q.to_lowercase();
    let node_match = block.get("nodes").and_then(|v| v.as_array()).is_some_and(|a| a.iter().any(|n| fm_has_match(n, q)));
    let self_match = ["name", "id"].iter().any(|k| block.get(*k).and_then(|v| v.as_str()).is_some_and(|s| s.to_lowercase().contains(&ql)));
    if node_match || self_match {
        let mut c = block.clone();
        c["sub_blocks"] = Value::Array(Vec::new());
        return Some(c);
    }
    None
}
