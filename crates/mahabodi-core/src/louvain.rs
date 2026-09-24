//! Deterministic port of fastmemory's in-process Louvain (`cluster::run_louvain_inline`,
//! github.com/fastBuilderAI/memory @ a7dec441, MIT).
//!
//! Same algorithm, same modularity-gain formula, same 20-sweep limit and strict-improvement
//! rule. The one change: fastmemory walks candidate communities through a `HashMap`, whose
//! iteration order is randomly seeded per process, so equal-gain ties (common on small,
//! symmetric memories) resolve differently run to run and the same memory gets different
//! communities. Here candidates are visited in ascending community id, so identical input
//! always yields identical communities (tested). `graph::ClusterEngine::FastMemory` still
//! calls fastmemory's own function for anyone who prefers it.

use std::collections::{BTreeMap, HashMap};

/// Returns (node ids in first-seen order, community index per node).
pub fn communities(edges: &[(String, String)]) -> (Vec<String>, Vec<usize>) {
    let mut idx: HashMap<&str, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    for (s, t) in edges {
        for n in [s.as_str(), t.as_str()] {
            if !idx.contains_key(n) {
                idx.insert(n, names.len());
                names.push(n.to_string());
            }
        }
    }
    let n = names.len();
    if n == 0 {
        return (names, Vec::new());
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut m = 0.0;
    for (s, t) in edges {
        let (a, b) = (idx[s.as_str()], idx[t.as_str()]);
        adj[a].push((b, 1.0));
        adj[b].push((a, 1.0));
        m += 1.0;
    }
    let degree: Vec<f64> = adj.iter().map(|v| v.iter().map(|(_, w)| w).sum()).collect();
    let mut community: Vec<usize> = (0..n).collect();
    let mut comm_total: Vec<f64> = degree.clone();

    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 20 {
        changed = false;
        iterations += 1;
        for i in 0..n {
            let current = community[i];
            let ki = degree[i];
            let mut nw: BTreeMap<usize, f64> = BTreeMap::new();
            for &(j, w) in &adj[i] {
                *nw.entry(community[j]).or_insert(0.0) += w;
            }
            let ki_in_current = nw.get(&current).copied().unwrap_or(0.0);
            let sigma_minus_i = comm_total[current] - ki;
            let remove_cost = ki_in_current / m - ki * sigma_minus_i / (2.0 * m * m);
            let (mut best, mut best_gain) = (current, 0.0);
            for (&c, &ki_in) in &nw {
                if c == current {
                    continue;
                }
                let gain = ki_in / m - ki * comm_total[c] / (2.0 * m * m) - remove_cost;
                if gain > best_gain {
                    best_gain = gain;
                    best = c;
                }
            }
            if best != current {
                comm_total[current] -= ki;
                comm_total[best] += ki;
                community[i] = best;
                changed = true;
            }
        }
    }
    // renumber communities densely in first-seen node order
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let dense = community
        .iter()
        .map(|c| {
            let k = remap.len();
            *remap.entry(*c).or_insert(k)
        })
        .collect();
    (names, dense)
}

/// Newman modularity of a partition (for comparing engines).
pub fn modularity(edges: &[(String, String)], names: &[String], comm: &[usize]) -> f64 {
    let pos: HashMap<&str, usize> = names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let m = edges.len() as f64;
    if m == 0.0 {
        return 0.0;
    }
    let mut deg = vec![0.0; names.len()];
    let mut inside: HashMap<usize, f64> = HashMap::new();
    for (s, t) in edges {
        let (a, b) = (pos[s.as_str()], pos[t.as_str()]);
        deg[a] += 1.0;
        deg[b] += 1.0;
        if comm[a] == comm[b] {
            *inside.entry(comm[a]).or_default() += 1.0;
        }
    }
    let mut tot: HashMap<usize, f64> = HashMap::new();
    for (i, d) in deg.iter().enumerate() {
        *tot.entry(comm[i]).or_default() += d;
    }
    tot.iter().map(|(c, t)| inside.get(c).copied().unwrap_or(0.0) / m - (t / (2.0 * m)).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn two_cliques_split() {
        let ed = e(&[("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z"), ("c", "x")]);
        let (names, c) = communities(&ed);
        let at = |s: &str| c[names.iter().position(|n| n == s).unwrap()];
        assert_eq!(at("a"), at("b"));
        assert_eq!(at("x"), at("z"));
        assert_ne!(at("a"), at("x"));
        assert!(modularity(&ed, &names, &c) > 0.3);
    }

    #[test]
    fn identical_input_identical_communities() {
        let ed = e(&[("F_a", "D_1"), ("F_b", "D_1"), ("F_c", "D_2"), ("F_d", "D_2"), ("F_a", "K_x"), ("F_c", "K_x")]);
        let first = communities(&ed);
        for _ in 0..50 {
            assert_eq!(communities(&ed), first);
        }
    }
}
