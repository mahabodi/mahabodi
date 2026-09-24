//! Concept-density guard.
//!
//! A fastmemory topology is only as searchable as its concept graph is dense: an
//! ATF with no data/access/event links is dropped by Louvain entirely, and one
//! with a single link is reachable from a single term. `measure` quantifies this
//! and `ensure` repairs it by linking each under-connected ATF to shared concept
//! nodes (`K_<stem>`) drawn from its own id/action/logic/text, then rebuilding
//! the fastmemory communities. Concepts are shared across ATFs, so they also
//! become the bridges Louvain clusters on.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::graph::Level;
use crate::memory::Memory;
use crate::query;
use crate::text;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DensityPolicy {
    /// Every ATF needs at least this many graph links.
    pub min_links_per_function: usize,
    pub max_isolated_functions: usize,
    /// Probe: querying an ATF's rarest own term must return that ATF in the top `probe_k`.
    pub min_probe_recall: f64,
    pub probe_k: usize,
    /// Probe at most this many ATFs (evenly strided) so large memories stay cheap to check.
    pub max_probes: usize,
    pub max_rounds: usize,
    /// Terms in more than this share of ATFs are too generic to be concepts (applies at >= 10 ATFs).
    pub max_concept_df_ratio: f64,
}

impl Default for DensityPolicy {
    fn default() -> Self {
        DensityPolicy {
            min_links_per_function: 2,
            max_isolated_functions: 0,
            min_probe_recall: 0.95,
            probe_k: 5,
            max_probes: 400,
            max_rounds: 3,
            max_concept_df_ratio: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DensityReport {
    pub functions: usize,
    pub nodes: usize,
    pub edges: usize,
    pub blocks: usize,
    pub concept_nodes: usize,
    pub isolated_functions: usize,
    pub min_links_per_function: usize,
    pub mean_links_per_function: f64,
    pub avg_degree: f64,
    pub singleton_block_ratio: f64,
    pub vocabulary: usize,
    pub probes: usize,
    pub probe_recall: f64,
    /// ATF ids whose probe failed (capped at 20).
    pub probe_failures: Vec<String>,
    pub passes: bool,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DensityOutcome {
    pub before: DensityReport,
    pub after: DensityReport,
    pub rounds: usize,
    pub concepts_added: usize,
}

/// The rarest content term of each ATF (the hardest honest query for it), with its document frequency.
fn probe_terms(m: &Memory) -> Vec<(String, String, usize)> {
    let g = m.graph();
    let ix = m.index();
    let mut out = Vec::new();
    for a in m.atfs() {
        let Some(&ni) = g.index_of.get(&format!("F_{}", a.id)) else { continue };
        let n = &g.nodes[ni];
        let mut ts: Vec<String> = text::terms(&format!("{} {}", n.label, n.text));
        ts.retain(|t| !t.chars().all(|c| c.is_ascii_digit()));
        ts.sort();
        ts.dedup();
        if let Some(t) = ts.into_iter().min_by_key(|t| (ix.doc_freq(t), t.clone())) {
            let df = ix.doc_freq(&t);
            out.push((a.id.clone(), t, df));
        }
    }
    out
}

pub fn measure(m: &Memory, p: &DensityPolicy) -> DensityReport {
    let g = m.graph();
    let funcs: Vec<usize> = g.functions().collect();
    let links: Vec<usize> = funcs.iter().map(|&f| g.degree(f)).collect();
    let isolated = links.iter().filter(|&&d| d == 0).count();
    let singletons = g.blocks.iter().filter(|b| b.members.len() == 1).count();

    // Probe only unambiguous terms (df <= k): with more holders than k, missing is not a density fault.
    let all = probe_terms(m);
    let stride = (all.len() / p.max_probes.max(1)).max(1);
    let mut probes = 0;
    let mut hits = 0;
    let mut failures = Vec::new();
    for (id, t, df) in all.iter().step_by(stride) {
        if *df > p.probe_k {
            continue;
        }
        probes += 1;
        let r = query::query(g, m.index(), t, p.probe_k);
        let target = format!("F_{id}");
        if r.matched && r.hits.iter().any(|h| h.id == target) {
            hits += 1;
        } else if failures.len() < 20 {
            failures.push(id.clone());
        }
    }
    let probe_recall = if probes == 0 { 1.0 } else { hits as f64 / probes as f64 };

    let min_links = links.iter().copied().min().unwrap_or(0);
    let mut violations = Vec::new();
    if funcs.is_empty() {
        violations.push("memory has no ATFs".to_string());
    }
    if isolated > p.max_isolated_functions {
        violations.push(format!("{isolated} isolated ATFs (max {})", p.max_isolated_functions));
    }
    if !funcs.is_empty() && min_links < p.min_links_per_function {
        let n = links.iter().filter(|&&d| d < p.min_links_per_function).count();
        violations.push(format!("{n} ATFs have < {} links", p.min_links_per_function));
    }
    if probe_recall < p.min_probe_recall {
        violations.push(format!("probe recall {:.3} < {:.3}", probe_recall, p.min_probe_recall));
    }
    DensityReport {
        functions: funcs.len(),
        nodes: g.nodes.len(),
        edges: g.edge_count,
        blocks: g.blocks.len(),
        concept_nodes: g.nodes.iter().filter(|n| n.level == Level::Concept).count(),
        isolated_functions: isolated,
        min_links_per_function: min_links,
        mean_links_per_function: if funcs.is_empty() { 0.0 } else { links.iter().sum::<usize>() as f64 / funcs.len() as f64 },
        avg_degree: if g.nodes.is_empty() { 0.0 } else { 2.0 * g.edge_count as f64 / g.nodes.len() as f64 },
        singleton_block_ratio: if g.blocks.is_empty() { 0.0 } else { singletons as f64 / g.blocks.len() as f64 },
        vocabulary: m.index().vocab_size(),
        probes,
        probe_recall,
        probe_failures: failures,
        passes: violations.is_empty(),
        violations,
    }
}

/// Densify until the policy passes or `max_rounds` is spent. Round r links each failing ATF
/// to up to 4 * 2^(r-1) concept stems from its own content, rarest-first.
pub fn ensure(m: &mut Memory, p: &DensityPolicy) -> DensityOutcome {
    let before = measure(m, p);
    let mut report = before.clone();
    let mut rounds = 0;
    let mut added = 0;
    while !report.passes && rounds < p.max_rounds && report.functions > 0 {
        rounds += 1;
        let cap = 4usize << (rounds - 1);
        let failing: HashSet<String> = report.probe_failures.iter().cloned().collect();

        // stem document frequency across ATFs
        let atf_terms: Vec<(String, Vec<String>)> = m
            .atfs
            .iter()
            .map(|a| {
                let body = format!(
                    "{} {} {} {} {}",
                    a.id, a.action, a.input, a.logic,
                    m.texts.get(&a.id).map(String::as_str).unwrap_or("")
                );
                let mut st: Vec<String> = text::terms(&body).iter().map(|t| text::stem(t)).collect();
                st.retain(|t| !t.chars().all(|c| c.is_ascii_digit()));
                (a.id.clone(), st)
            })
            .collect();
        let mut df: HashMap<&str, usize> = HashMap::new();
        for (_, st) in &atf_terms {
            let uniq: HashSet<&str> = st.iter().map(String::as_str).collect();
            for t in uniq {
                *df.entry(t).or_default() += 1;
            }
        }
        let n = atf_terms.len();
        let too_common = |t: &str| n >= 10 && df[t] as f64 / n as f64 > p.max_concept_df_ratio;

        let g = m.graph();
        let have: HashSet<(String, String)> = m.concepts.iter().cloned().collect();
        let mut new_edges = Vec::new();
        for (id, st) in &atf_terms {
            let deg = g.index_of.get(&format!("F_{id}")).map_or(0, |&i| g.degree(i));
            let needs = deg < p.min_links_per_function.max(1) || failing.contains(id) || rounds > 1 && deg < cap;
            if !needs {
                continue;
            }
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in st {
                *tf.entry(t.as_str()).or_default() += 1;
            }
            let mut cands: Vec<&str> = tf.keys().copied().filter(|t| !too_common(t)).collect();
            if cands.is_empty() {
                cands = tf.keys().copied().collect(); // generic terms beat no links at all
            }
            // shared-but-rare first (bridges), then own-only; ties by in-ATF frequency
            cands.sort_by_key(|t| (df[t] <= 1, df[t], std::cmp::Reverse(tf[t]), t.to_string()));
            for t in cands.into_iter().take(cap) {
                let e = (id.clone(), t.to_string());
                if !have.contains(&e) {
                    new_edges.push(e);
                }
            }
        }
        new_edges.sort();
        new_edges.dedup();
        if new_edges.is_empty() {
            break;
        }
        added += new_edges.len();
        m.concepts.extend(new_edges);
        m.rebuild();
        report = measure(m, p);
    }
    DensityOutcome { before, after: report, rounds, concepts_added: added }
}
