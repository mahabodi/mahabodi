//! Inverted index over graph nodes: exact terms, stems, and a character-trigram
//! map over the vocabulary for typo-tolerant term lookup. Scoring is BM25.

use std::collections::{HashMap, HashSet};

use crate::graph::{Graph, Level};
use crate::text;

#[derive(Debug, Default, Clone)]
pub struct Index {
    exact: HashMap<String, Vec<(usize, f32)>>,
    stems: HashMap<String, Vec<(usize, f32)>>,
    doc_len: Vec<f32>,
    avg_len: f32,
    n_docs: usize,
    vocab: Vec<String>,
    tri: HashMap<String, Vec<usize>>,
}

/// Field weights: a term in a node's id/label says more than one in its passage.
const W_LABEL: f32 = 3.0;
const W_TEXT: f32 = 1.0;

impl Index {
    pub fn build(g: &Graph) -> Index {
        use rayon::prelude::*;
        let mut ix = Index { n_docs: g.nodes.len(), doc_len: vec![0.0; g.nodes.len()], ..Default::default() };
        // Tokenising and stemming is per node and independent: done in parallel. The postings are then
        // merged in node order, so every posting list is identical to a sequential build.
        let per_node: Vec<(f32, HashMap<String, f32>, HashMap<String, f32>)> = g
            .nodes
            .par_iter()
            .map(|n| {
                let mut tf: HashMap<String, f32> = HashMap::new();
                for t in text::terms(&n.label) {
                    *tf.entry(t).or_default() += W_LABEL;
                }
                for t in text::terms(&n.text) {
                    *tf.entry(t).or_default() += W_TEXT;
                }
                let len = tf.values().sum();
                let mut stf: HashMap<String, f32> = HashMap::new();
                for (t, w) in &tf {
                    *stf.entry(text::stem(t)).or_default() += *w;
                }
                (len, tf, stf)
            })
            .collect();
        for (i, (len, tf, stf)) in per_node.into_iter().enumerate() {
            ix.doc_len[i] = len;
            for (t, w) in tf {
                ix.exact.entry(t).or_default().push((i, w));
            }
            for (t, w) in stf {
                ix.stems.entry(t).or_default().push((i, w));
            }
        }
        ix.avg_len = if ix.n_docs > 0 { ix.doc_len.iter().sum::<f32>() / ix.n_docs as f32 } else { 0.0 };
        ix.vocab = ix.exact.keys().cloned().collect();
        ix.vocab.par_sort();
        let grams: Vec<std::collections::HashSet<String>> = ix.vocab.par_iter().map(|w| text::trigrams(w)).collect();
        for (vi, gs) in grams.into_iter().enumerate() {
            for g in gs {
                ix.tri.entry(g).or_default().push(vi);
            }
        }
        ix
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn has_term(&self, t: &str) -> bool {
        self.exact.contains_key(t)
    }

    pub fn doc_freq(&self, t: &str) -> usize {
        self.exact.get(t).map_or(0, |p| p.len())
    }

    fn bm25(&self, postings: &[(usize, f32)], weight: f32, scores: &mut HashMap<usize, f32>) {
        let (k1, b) = (1.2f32, 0.75f32);
        let df = postings.len() as f32;
        let idf = ((self.n_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
        for &(d, tf) in postings {
            let norm = k1 * (1.0 - b + b * self.doc_len[d] / self.avg_len.max(1e-6));
            *scores.entry(d).or_default() += weight * idf * tf * (k1 + 1.0) / (tf + norm);
        }
    }

    /// BM25 over exact terms. Returns (node, score), unsorted.
    pub fn search_exact(&self, terms: &[String]) -> HashMap<usize, f32> {
        let mut s = HashMap::new();
        for t in dedup(terms) {
            if let Some(p) = self.exact.get(&t) {
                self.bm25(p, 1.0, &mut s);
            }
        }
        s
    }

    pub fn search_stem(&self, terms: &[String]) -> HashMap<usize, f32> {
        let mut s = HashMap::new();
        for t in dedup(terms) {
            if let Some(p) = self.stems.get(&text::stem(&t)) {
                self.bm25(p, 1.0, &mut s);
            }
        }
        s
    }

    /// Vocabulary terms within trigram-Jaccard `min_sim` of `t`, best first (max `k`).
    pub fn similar_terms(&self, t: &str, min_sim: f64, k: usize) -> Vec<(String, f64)> {
        let q = text::trigrams(t);
        let mut cand: HashMap<usize, usize> = HashMap::new();
        for g in &q {
            for &vi in self.tri.get(g).map(|v| v.as_slice()).unwrap_or(&[]) {
                *cand.entry(vi).or_default() += 1;
            }
        }
        let mut out: Vec<(String, f64)> = cand
            .into_iter()
            .map(|(vi, inter)| {
                let w = &self.vocab[vi];
                let wl = text::trigrams(w).len();
                (w.clone(), inter as f64 / (q.len() + wl - inter) as f64)
            })
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.truncate(k);
        out
    }

    /// Typo-tolerant search: each query term expands to similar vocabulary terms, weighted by similarity.
    pub fn search_fuzzy(&self, terms: &[String], min_sim: f64) -> (HashMap<usize, f32>, Vec<(String, String, f64)>) {
        let mut s = HashMap::new();
        let mut used = Vec::new();
        for t in dedup(terms) {
            for (w, sim) in self.similar_terms(&t, min_sim, 3) {
                if let Some(p) = self.exact.get(&w) {
                    self.bm25(p, sim as f32, &mut s);
                    used.push((t.clone(), w, sim));
                }
            }
        }
        (s, used)
    }

    /// Function nodes ranked by degree: the "most connected memory" used as last-resort context.
    pub fn hubs(g: &Graph, k: usize) -> Vec<usize> {
        let mut f: Vec<usize> = g.functions().collect();
        if f.is_empty() {
            f = (0..g.nodes.len()).collect();
        }
        f.sort_by(|&a, &b| g.degree(b).cmp(&g.degree(a)).then(g.nodes[a].id.cmp(&g.nodes[b].id)));
        f.truncate(k);
        f
    }
}

fn dedup(terms: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    terms.iter().filter(|t| seen.insert(t.as_str())).cloned().collect()
}

/// Levels that carry retrievable content, used to prefer them in result lists.
pub fn level_rank(l: Level) -> u8 {
    match l {
        Level::Function => 0,
        Level::Data => 1,
        Level::Concept => 2,
        Level::Event => 3,
        Level::Access => 4,
    }
}
