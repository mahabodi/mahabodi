//! Bodi's decision layer on top of the Laya checkpoint.
//!
//! Same model, same weights, different use of it:
//!
//! * **Tournament shortlisting** - Laya gives all options of a question one shared
//!   `head_max_len` token budget, so beyond ~20 options each label keeps ~4 tokens and
//!   they stop being distinguishable (Laya's banking77 result: 0.425 at 77 labels).
//!   Bodi scores options in groups of at most `max_options_per_pass`, keeps the top
//!   `finalists_per_group` of each, and decides among the finalists in a final round.
//! * **Order ensemble** - every row is also scored with its options reversed and the
//!   probabilities averaged, to reduce option-order sensitivity (Laya documents position
//!   bias for its multilingual checkpoint on score questions; the effect on the English
//!   checkpoint is measured, not assumed: see research/bench.py order-flip rate).
//! * **Script gate** - an English-only checkpoint is never run on text it cannot read;
//!   the answer comes back with a null decision and `handoff: true`.
//! * **Cache** - identical (state, question, options) decisions are answered from memory.
//!
//! All rows of a round, across every state and question of a call, go through the model
//! together (length-bucketed), so the extra rows cost batched, not sequential, passes.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::runtime::{answer_json_with, LayaModel};
use super::script;
use super::sequence::{self, py_json, Encoded, QType, Question};
use super::{confidence_from_probs, softmax_t};
use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DecideOptions {
    /// Average each row with its option-reversed twin.
    pub permute: bool,
    /// Adaptive ensemble: only add the reversed twin when the single-order confidence is
    /// below this (0 = always add it when `permute`). Tuned on validation data, never test.
    pub permute_below_confidence: f64,
    /// Also ensemble tournament group rounds (default: only the final round).
    pub permute_groups: bool,
    /// Ensemble the final round of shortlisted (tournament) questions even when `permute` is
    /// off. Validation (banking77 train, N=300): 16/3 tournament 0.6133 -> 0.6467 with it.
    pub permute_shortlisted_final: bool,
    /// Experience memory: nearest labelled past cases consulted per decision (0 disables).
    /// Active only for questions with stored experience (`learn`) and a model exported with
    /// the `pooled` output. log p = log p_laya + experience_weight * log(p_mem + 1e-3).
    pub experience_k: usize,
    pub experience_weight: f64,
    /// Softmax temperature over cosine similarities of the neighbours.
    pub experience_temperature: f64,
    /// Tournament questions: the memory's top-N labels join the final round.
    pub experience_candidates: usize,
    /// Scale experience_weight by the memory's self-measured trust: leave-one-out kNN accuracy
    /// over the stored cases relative to the majority-class rate, clamped to [0, 1]. Memory whose
    /// neighbours predict no better than always answering the commonest label gets weight 0.
    pub experience_auto_trust: bool,
    /// Margin gate: memory only changes an answer when Laya's final-round top-1 minus top-2
    /// probability is below this (1.0+ = always). Keeps confident Laya answers untouched; tuned on
    /// validation (research/results/tune_margin_gate.json: no suite below Laya, mean +3.2 pts).
    pub experience_below_margin: f64,
    /// Agreement override (0 = off): memory replaces even a confident Laya answer when at least this
    /// many of the `experience_k` nearest stored cases share one label AND the task's memory is
    /// reliable (leave-one-out vote accuracy of the stored cases >= experience_override_min_trust).
    /// The answer is then the neighbours' vote. Tuned on validation: research/results/tune_agree_gate.json.
    pub experience_override_agree: usize,
    pub experience_override_min_trust: f64,
    /// Zero-shot embedding shortlist for many-option choice questions (needs an embedder):
    /// keep the N options most similar to the state text before Laya scores them (0 = off).
    pub embed_shortlist: usize,
    /// Instead of replacing the tournament, add the embedding shortlist to its finalists.
    pub embed_shortlist_union: bool,
    /// Choice questions with more options than this are shortlisted (0 = never).
    pub max_options_per_pass: usize,
    pub finalists_per_group: usize,
    /// Answers under this entropy-confidence are flagged `handoff`. Default 0.0: System 1
    /// never escalates on confidence unless the caller sets a threshold (only the script
    /// gate hands off by default). Probabilities after ensembling/shortlisting are not
    /// what Laya's temperatures were fitted for; calibration is measured in the benchmark.
    pub min_confidence: f64,
    pub script_gate: bool,
    pub cache: bool,
    /// Round numbers to 4 decimals like Laya's schema (default). Off returns exact probabilities.
    pub round_probabilities: bool,
}

impl Default for DecideOptions {
    fn default() -> Self {
        // permute off by default: on validation data (research/results/tune.json, N=300) the
        // order ensemble changed no ag_news decision and cost -0.3 pt on emotion at 2x rows.
        DecideOptions {
            permute: false,
            permute_below_confidence: 0.0,
            permute_groups: false,
            permute_shortlisted_final: true,
            // validation-tuned global defaults with the margin gate (tune_margin_gate.json)
            experience_k: 8,
            experience_weight: 2.0,
            experience_temperature: 0.2,
            experience_candidates: 3,
            experience_auto_trust: false,
            experience_below_margin: 0.5,
            experience_override_agree: 0,
            experience_override_min_trust: 0.6,
            embed_shortlist: 0,
            embed_shortlist_union: false,
            max_options_per_pass: 16,
            finalists_per_group: 3,
            min_confidence: 0.0,
            script_gate: true,
            cache: true,
            round_probabilities: true,
        }
    }
}

impl DecideOptions {
    /// Plain Laya behaviour (for parity checks and A/B benchmarks).
    pub fn laya_compatible() -> Self {
        DecideOptions { permute: false, max_options_per_pass: 0, script_gate: false, cache: false, experience_k: 0, ..Default::default() }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Stats {
    pub decisions: u64,
    pub cache_hits: u64,
    pub rows_scored: u64,
    pub forward_rounds: u64,
    pub script_gated: u64,
}

/// Least-recently-used answer cache keyed by the full (state, question, options) string, so a
/// hash collision can never return another question's answer. `tick` stamps recency; eviction
/// drops the stalest stamp (lazy deletion of superseded stamps in the queue).
struct Lru {
    map: HashMap<String, (Value, u64)>,
    order: VecDeque<(u64, String)>,
    tick: u64,
    cap: usize,
}

impl Lru {
    fn get(&mut self, k: &str) -> Option<Value> {
        self.tick += 1;
        let t = self.tick;
        let (v, stamp) = self.map.get_mut(k)?;
        *stamp = t;
        self.order.push_back((t, k.to_string()));
        Some(v.clone())
    }
    fn put(&mut self, k: String, v: Value) {
        self.tick += 1;
        self.map.insert(k.clone(), (v, self.tick));
        self.order.push_back((self.tick, k));
        while self.map.len() > self.cap {
            let Some((t, old)) = self.order.pop_front() else { break };
            if self.map.get(&old).is_some_and(|(_, s)| *s == t) {
                self.map.remove(&old);
            }
        }
        if self.order.len() > 4 * self.cap.max(1) {
            let live: std::collections::HashSet<(u64, &String)> = self.map.iter().map(|(k, (_, t))| (*t, k)).collect();
            let keep: VecDeque<(u64, String)> = self.order.iter().filter(|(t, k)| live.contains(&(*t, k))).cloned().collect();
            self.order = keep;
        }
    }
}

/// Labelled past cases per question definition: (unit-norm vector, option index).
type Examples = std::sync::Arc<Vec<(Vec<f32>, usize)>>;

/// Leave-one-out estimates probe at most this many stored cases (evenly strided), each against
/// ALL stored cases: cost O(min(n, LOO_PROBES) * n * dim) per learn() call on that question.
/// Memories of up to LOO_PROBES cases get the exact full leave-one-out value.
pub const LOO_PROBES: usize = 2000;

fn loo_probes(n: usize) -> Vec<usize> {
    if n <= LOO_PROBES {
        (0..n).collect()
    } else {
        (0..LOO_PROBES).map(|i| i * n / LOO_PROBES).collect()
    }
}

/// Self-measured trust of a memory: leave-one-out kNN accuracy (k=8, T=0.2) against the
/// majority-class rate: (loo - maj) / (1 - maj), clamped to [0, 1]. Uses only stored cases.
pub fn memory_trust(ex: &[(Vec<f32>, usize)], n_options: usize) -> f64 {
    let n = ex.len();
    if n < 4 {
        return 0.0;
    }
    let mut counts = vec![0usize; n_options.max(1)];
    for (_, l) in ex {
        if *l < counts.len() {
            counts[*l] += 1;
        }
    }
    let maj = *counts.iter().max().unwrap() as f64 / n as f64;
    if maj >= 1.0 {
        return 0.0;
    }
    let probes = loo_probes(n);
    let mut correct = 0usize;
    for &i in &probes {
        let mut sims: Vec<(f32, usize)> = (0..n).filter(|&j| j != i)
            .map(|j| (ex[i].0.iter().zip(&ex[j].0).map(|(a, b)| a * b).sum::<f32>(), ex[j].1)).collect();
        let k = 8.min(sims.len());
        sims.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top = &sims[..k];
        let smax = top.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut p = vec![0.0f64; n_options.max(1)];
        for (sim, l) in top {
            if *l < p.len() {
                p[*l] += ((*sim as f64 - smax) / 0.2).exp();
            }
        }
        let best = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
        correct += (best == ex[i].1) as usize;
    }
    let loo = correct as f64 / probes.len() as f64;
    ((loo - maj) / (1.0 - maj)).clamp(0.0, 1.0)
}

/// Leave-one-out accuracy of an unweighted k-nearest-neighbour vote over the stored cases (ties go
/// to the lowest option index). Uses only stored cases, so it is available at decision time.
pub fn loo_vote_accuracy(ex: &[(Vec<f32>, usize)], n_options: usize, k: usize) -> f64 {
    let n = ex.len();
    if n < 2 || k == 0 {
        return 0.0;
    }
    let probes = loo_probes(n);
    let mut correct = 0usize;
    for &i in &probes {
        let v = votes_excluding(&ex[i].0, ex, n_options, k, Some(i));
        correct += (argmax_votes(&v) == ex[i].1) as usize;
    }
    correct as f64 / probes.len() as f64
}

fn argmax_votes(v: &[usize]) -> usize {
    let mut best = 0;
    for (i, &c) in v.iter().enumerate() {
        if c > v[best] {
            best = i;
        }
    }
    best
}

/// Unweighted label counts among the k stored cases most similar to `emb` (optionally skipping one).
fn votes_excluding(emb: &[f32], ex: &[(Vec<f32>, usize)], n: usize, k: usize, skip: Option<usize>) -> Vec<usize> {
    let e = unit(emb);
    let mut sims: Vec<(f32, usize)> = ex.iter().enumerate().filter(|(i, _)| Some(*i) != skip)
        .map(|(_, (v, l))| (v.iter().zip(&e).map(|(a, b)| a * b).sum::<f32>(), *l)).collect();
    let k = k.min(sims.len());
    let mut out = vec![0usize; n.max(1)];
    if k == 0 {
        return out;
    }
    sims.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (_, l) in &sims[..k] {
        if *l < out.len() {
            out[*l] += 1;
        }
    }
    out
}

pub struct System1 {
    pub model: LayaModel,
    cache: Mutex<Lru>,
    stats: Mutex<Stats>,
    experience: std::sync::RwLock<HashMap<String, Examples>>,
    /// memory_trust per experience key, recomputed on every learn.
    trust: std::sync::RwLock<HashMap<String, f64>>,
    /// loo_vote_accuracy (k=8) per experience key, recomputed on every learn (agreement override).
    loo: std::sync::RwLock<HashMap<String, f64>>,
    /// Dense text embedder for experience memory; without it the model's pooled vector is used.
    embedder: std::sync::RwLock<Option<std::sync::Arc<super::embedder::Embedder>>>,
    /// Rendered-option embeddings per question definition (embedding shortlist).
    option_emb: std::sync::RwLock<HashMap<String, std::sync::Arc<Vec<Vec<f32>>>>>,
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// Label distribution over `n` options from the k most similar stored cases.
fn knn(emb: &[f32], ex: &[(Vec<f32>, usize)], n: usize, k: usize, temp: f64) -> (Vec<f64>, usize) {
    let e = unit(emb);
    let mut sims: Vec<(f32, usize)> = ex.iter().map(|(v, l)| (v.iter().zip(&e).map(|(a, b)| a * b).sum::<f32>(), *l)).collect();
    let k = k.min(sims.len());
    sims.select_nth_unstable_by(k.saturating_sub(1), |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top = &sims[..k];
    let smax = top.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p = vec![0.0; n];
    for (sim, l) in top {
        if *l < n {
            p[*l] += ((*sim as f64 - smax) / temp.max(1e-6)).exp();
        }
    }
    let z: f64 = p.iter().sum::<f64>().max(1e-12);
    (p.into_iter().map(|x| x / z).collect(), k)
}

/// Parse a gold label for `q` (choice: label string; score: level index; noul: bool/"true"/1).
fn label_index(q: &Question, v: &Value) -> Result<usize> {
    let bad = || crate::error::BodiError::Invalid(format!("question {:?}: bad label {v}", q.id));
    match q.t {
        QType::Choice => {
            let s = v.as_str().map(String::from).unwrap_or_else(|| v.to_string());
            q.choice_labels().iter().position(|l| *l == s).ok_or_else(bad)
        }
        QType::Score => v.as_u64().map(|x| x as usize).filter(|&x| x < q.n_options()).ok_or_else(bad),
        QType::Noul => match v {
            Value::Bool(b) => Ok(*b as usize),
            Value::Number(n) if n.as_u64() == Some(0) || n.as_u64() == Some(1) => Ok(n.as_u64().unwrap() as usize),
            Value::String(s) if s.eq_ignore_ascii_case("true") => Ok(1),
            Value::String(s) if s.eq_ignore_ascii_case("false") => Ok(0),
            _ => Err(bad()),
        },
    }
}

/// One pending (state, question) decision being worked through rounds.
struct Job {
    state: usize,
    q: usize,
    /// Candidate option indices (into the question's full option list) still alive.
    alive: Vec<usize>,
    shortlisted: bool,
    rounds: usize,
    final_p: Option<Vec<f64>>,
    act: f64,
    /// Adaptive ensemble: the single-order final result awaiting its reversed twin.
    pending_twin: Option<Vec<f64>>,
    twinned: bool,
    /// Experience memory label distribution (set after the first round) and neighbours used.
    p_mem: Option<(Vec<f64>, usize)>,
    /// Unweighted neighbour label counts (agreement override).
    votes: Option<Vec<usize>>,
    overridden: bool,
    /// Embedding-shortlist candidates to merge into the tournament final (union mode).
    emb_top: Vec<usize>,
}


impl System1 {
    pub fn new(model: LayaModel) -> System1 {
        System1 {
            model,
            cache: Mutex::new(Lru { map: HashMap::new(), order: VecDeque::new(), tick: 0, cap: 16384 }),
            stats: Mutex::new(Stats::default()),
            experience: std::sync::RwLock::new(HashMap::new()),
            trust: std::sync::RwLock::new(HashMap::new()),
            loo: std::sync::RwLock::new(HashMap::new()),
            embedder: std::sync::RwLock::new(None),
            option_emb: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn set_embedder(&self, e: Option<std::sync::Arc<super::embedder::Embedder>>) {
        *self.embedder.write().unwrap() = e;
    }

    fn embedder(&self) -> Option<std::sync::Arc<super::embedder::Embedder>> {
        self.embedder.read().unwrap().clone()
    }

    /// Experience keys carry the embedding space so text and pooled vectors never mix.
    fn exp_key(&self, qdef: &str) -> String {
        format!("{}|{qdef}", if self.embedder().is_some() { "text" } else { "pooled" })
    }

    /// Store labelled cases as experience. `labels[i]` maps question id -> gold label for
    /// `states[i]`. Costs one forward row per (state, question). Requires the `pooled` output.
    pub fn learn(&self, states: &[Value], questions: &Value, labels: &[Value]) -> Result<Value> {
        if states.len() != labels.len() {
            return Err(crate::error::BodiError::Invalid("states and labels differ in length".into()));
        }
        let qs = sequence::parse_questions(questions)?;
        let qdefs: Vec<String> = questions.as_object().map(|m| m.values().map(py_json).collect()).unwrap_or_default();
        let mut rows = Vec::new();
        let mut texts = Vec::new();
        let mut meta = Vec::new();
        for (si, st) in states.iter().enumerate() {
            for (qi, q) in qs.iter().enumerate() {
                let Some(lv) = labels[si].get(&q.id) else { continue };
                meta.push((qi, label_index(q, lv)?));
                texts.push(sequence::plain_text(st));
                rows.push(self.model.encode(st, q, None));
            }
        }
        let vecs: Vec<Vec<f32>> = match self.embedder() {
            // text space: embed the state only; no Laya forward pass needed
            Some(e) => e.embed(&texts)?,
            None => {
                let refs: Vec<&Encoded> = rows.iter().collect();
                self.model.run(&refs)?.into_iter().map(|r| r.pooled.ok_or_else(|| crate::error::BodiError::Model(
                    "this model export has no `pooled` output; re-run research/export_onnx.py or load an embedder".into()))).collect::<Result<_>>()?
            }
        };
        let mut add: HashMap<usize, Vec<(Vec<f32>, usize)>> = HashMap::new();
        for ((qi, l), v) in meta.into_iter().zip(vecs) {
            add.entry(qi).or_default().push((unit(&v), l));
        }
        let mut store = self.experience.write().unwrap();
        let mut counts = Map::new();
        let mut trusts = Map::new();
        for (qi, new) in add {
            let key = self.exp_key(&qdefs[qi]);
            let e = store.entry(key.clone()).or_default();
            let mut v = (**e).clone();
            v.extend(new);
            let t = memory_trust(&v, qs[qi].n_options());
            self.loo.write().unwrap().insert(key.clone(), loo_vote_accuracy(&v, qs[qi].n_options(), 8));
            self.trust.write().unwrap().insert(key, t);
            counts.insert(qs[qi].id.clone(), json!(v.len()));
            trusts.insert(qs[qi].id.clone(), json!(super::round4(t)));
            *e = std::sync::Arc::new(v);
        }
        Ok(json!({"stored": counts, "trust": trusts}))
    }

    /// The pooled decision-context vector for every (state, question) pair: states x questions.
    pub fn embed(&self, states: &[Value], questions: &Value) -> Result<Vec<Vec<Vec<f32>>>> {
        let qs = sequence::parse_questions(questions)?;
        let rows: Vec<Encoded> = states.iter().flat_map(|st| qs.iter().map(move |q| (st, q))).map(|(st, q)| self.model.encode(st, q, None)).collect();
        let refs: Vec<&Encoded> = rows.iter().collect();
        let scored = self.model.run(&refs)?;
        let mut it = scored.into_iter();
        let mut out = Vec::with_capacity(states.len());
        for _ in states {
            let mut per = Vec::with_capacity(qs.len());
            for _ in &qs {
                per.push(it.next().and_then(|r| r.pooled).ok_or_else(|| crate::error::BodiError::Model(
                    "this model export has no `pooled` output; re-run research/export_onnx.py".into()))?);
            }
            out.push(per);
        }
        Ok(out)
    }

    pub fn forget(&self) {
        self.experience.write().unwrap().clear();
        self.trust.write().unwrap().clear();
        self.loo.write().unwrap().clear();
    }

    pub fn experience_size(&self) -> usize {
        self.experience.read().unwrap().values().map(|v| v.len()).sum()
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }

    pub fn decide(&self, state: &Value, questions: &Value, opts: &DecideOptions) -> Result<Value> {
        Ok(self.decide_batch(std::slice::from_ref(state), questions, opts)?.remove(0))
    }

    pub fn decide_batch(&self, states: &[Value], questions: &Value, opts: &DecideOptions) -> Result<Vec<Value>> {
        let qs = sequence::parse_questions(questions)?;
        let qdefs: Vec<String> = questions.as_object().map(|m| m.values().map(py_json).collect()).unwrap_or_default();
        // every option that can change an answer is part of the key
        let opt_key = serde_json::to_string(&DecideOptions { cache: true, ..opts.clone() }).unwrap_or_default();
        let exp: Vec<Option<Examples>> = {
            let store = self.experience.read().unwrap();
            qdefs.iter().map(|d| if opts.experience_k > 0 { store.get(&self.exp_key(d)).cloned().filter(|e| !e.is_empty()) } else { None }).collect()
        };
        let exp_w: Vec<f64> = {
            let t = self.trust.read().unwrap();
            qdefs.iter().map(|d| {
                let tr = if opts.experience_auto_trust { t.get(&self.exp_key(d)).copied().unwrap_or(0.0) } else { 1.0 };
                opts.experience_weight * tr
            }).collect()
        };
        let exp_loo: Vec<f64> = {
            let t = self.loo.read().unwrap();
            qdefs.iter().map(|d| t.get(&self.exp_key(d)).copied().unwrap_or(0.0)).collect()
        };
        // stored experience changes answers, so its size is part of the cache key
        let opt_key = format!("{opt_key}|exp={}", exp.iter().map(|e| e.as_ref().map_or(0, |v| v.len())).sum::<usize>());
        let mut out: Vec<Map<String, Value>> = vec![Map::new(); states.len()];
        let mut tokens = vec![0usize; states.len()];
        let mut jobs: Vec<Job> = Vec::new();
        let mut stats = Stats::default();

        for (si, st) in states.iter().enumerate() {
            let st_text = sequence::serialize_state(st);
            let gated = opts.script_gate && self.model.english_only && {
                let all = format!("{st_text} {}", qs.iter().map(|q| q.ins.as_str()).collect::<Vec<_>>().join(" "));
                script::unreadable_for_english(&all)
            };
            for (qi, q) in qs.iter().enumerate() {
                stats.decisions += 1;
                if gated {
                    stats.script_gated += 1;
                    let n = q.n_options();
                    let mut a = answer_json_with(q, &vec![1.0 / n as f64; n], 0.0, opts.round_probabilities);
                    // no decision: a caller reading choice/score/noul must not act on a label
                    for k in ["choice", "score", "noul"] {
                        if a.get(k).is_some() {
                            a[k] = Value::Null;
                        }
                    }
                    a["confidence"] = json!(0.0);
                    a["bodi"] = json!({"handoff": true, "out_of_distribution": true, "strategy": "script_gate",
                        "reason": format!("{} script: this checkpoint reads English/Latin text only", script::profile(&st_text).script)});
                    out[si].insert(q.id.clone(), a);
                    continue;
                }
                if opts.cache {
                    let key = format!("{st_text}\u{1}{}\u{1}{opt_key}", qdefs[qi]);
                    if let Some(mut hit) = self.cache.lock().unwrap().get(&key) {
                        stats.cache_hits += 1;
                        hit["bodi"]["cached"] = json!(true);
                        out[si].insert(q.id.clone(), hit);
                        continue;
                    }
                }
                let n = q.n_options();
                let shortlisted = q.t == QType::Choice && opts.max_options_per_pass > 0 && n > opts.max_options_per_pass;
                jobs.push(Job { state: si, q: qi, alive: (0..n).collect(), shortlisted, rounds: 0, final_p: None, act: 0.0, pending_twin: None, twinned: false, p_mem: None, votes: None, overridden: false, emb_top: Vec::new() });
            }
        }

        // Text space (embedder loaded): embed each needed state once, up front. Used for
        // experience memory (p_mem) and for the zero-shot embedding shortlist.
        let text_embedder = self.embedder();
        if let Some(e) = &text_embedder {
            let wants_short = |j: &Job| j.shortlisted && opts.embed_shortlist > 0;
            let need: Vec<usize> = jobs.iter().enumerate().filter(|(_, j)| exp[j.q].is_some() || wants_short(j)).map(|(i, _)| i).collect();
            if !need.is_empty() {
                let mut st_idx: Vec<usize> = need.iter().map(|&i| jobs[i].state).collect();
                st_idx.sort_unstable();
                st_idx.dedup();
                let texts: Vec<String> = st_idx.iter().map(|&s| sequence::plain_text(&states[s])).collect();
                let vecs = e.embed(&texts)?;
                for i in need {
                    let v = &vecs[st_idx.binary_search(&jobs[i].state).unwrap()];
                    let q = &qs[jobs[i].q];
                    if let Some(ex) = &exp[jobs[i].q] {
                        jobs[i].p_mem = Some(knn(v, ex, q.n_options(), opts.experience_k, opts.experience_temperature));
                        jobs[i].votes = Some(votes_excluding(v, ex, q.n_options(), opts.experience_k, None));
                    }
                    if wants_short(&jobs[i]) {
                        let oe = {
                            let cached = self.option_emb.read().unwrap().get(&qdefs[jobs[i].q]).cloned();
                            match cached {
                                Some(c) => c,
                                None => {
                                    let c = std::sync::Arc::new(e.embed(&sequence::render_options(q))?);
                                    self.option_emb.write().unwrap().insert(qdefs[jobs[i].q].clone(), c.clone());
                                    c
                                }
                            }
                        };
                        let mut idx: Vec<(f32, usize)> = oe.iter().enumerate().map(|(o, w)| (w.iter().zip(v).map(|(a, b)| a * b).sum(), o)).collect();
                        idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                        let mut top: Vec<usize> = idx.into_iter().take(opts.embed_shortlist).map(|x| x.1).collect();
                        top.sort_unstable();
                        if opts.embed_shortlist_union {
                            jobs[i].emb_top = top;
                        } else {
                            jobs[i].alive = top; // Laya decides among the embedding shortlist
                        }
                    }
                }
            }
        }

        // Rounds: every unfinished job contributes its rows; one batched model pass per round.
        loop {
            let mut rows: Vec<Encoded> = Vec::new();
            let mut plan: Vec<(usize, Vec<(Vec<usize>, Vec<usize>)>)> = Vec::new(); // job -> [(group option idx, row ids)]
            let mut embed_row: HashMap<usize, usize> = HashMap::new(); // job -> row giving its pooled vector
            for (ji, j) in jobs.iter().enumerate() {
                if j.final_p.is_some() {
                    continue;
                }
                let q = &qs[j.q];
                if j.rounds == 0 && exp[j.q].is_some() && j.shortlisted && text_embedder.is_none() {
                    // the full-question row (Laya's own sequence) is the case's embedding
                    embed_row.insert(ji, rows.len());
                    rows.push(self.model.encode(&states[j.state], q, None));
                }
                let groups: Vec<Vec<usize>> = if j.shortlisted && j.alive.len() > opts.max_options_per_pass {
                    let g = j.alive.len().div_ceil(opts.max_options_per_pass);
                    let size = j.alive.len().div_ceil(g);
                    j.alive.chunks(size).map(|c| c.to_vec()).collect()
                } else {
                    vec![j.alive.clone()]
                };
                let final_round = groups.len() == 1;
                let adaptive = opts.permute_below_confidence > 0.0;
                let mut gplan = Vec::new();
                for grp in groups {
                    let sub = if q.t == QType::Choice && grp.len() != q.n_options() { q.with_choice_subset(&grp) } else { q.clone() };
                    let k = grp.len();
                    let rev: Vec<usize> = (0..k).rev().collect();
                    let mut ids = Vec::new();
                    if j.pending_twin.is_some() {
                        // second pass of an adaptive ensemble: only the reversed row
                        ids.push(rows.len());
                        rows.push(self.model.encode(&states[j.state], &sub, Some(&rev)));
                    } else {
                        ids.push(rows.len());
                        rows.push(self.model.encode(&states[j.state], &sub, None));
                        let wants = opts.permute || (j.shortlisted && opts.permute_shortlisted_final && final_round);
                        let twin_now = wants && k >= 2 && !adaptive && (final_round || opts.permute_groups);
                        if twin_now {
                            ids.push(rows.len());
                            rows.push(self.model.encode(&states[j.state], &sub, Some(&rev)));
                        }
                    }
                    gplan.push((grp, ids));
                }
                plan.push((ji, gplan));
            }
            if rows.is_empty() {
                break;
            }
            for (ji, gplan) in &plan {
                let q = &qs[jobs[*ji].q];
                for (grp, ids) in gplan {
                    if rows[ids[0]].markers.len() != grp.len() {
                        return Err(crate::error::BodiError::Invalid(format!(
                            "question {:?}: only {} of {} option markers fit in max_len={} (set max_options_per_pass to shortlist)",
                            q.id, rows[ids[0]].markers.len(), grp.len(), self.model.max_len)));
                    }
                }
            }
            let refs: Vec<&Encoded> = rows.iter().collect();
            let scored = self.model.run(&refs)?;
            stats.rows_scored += rows.len() as u64;
            stats.forward_rounds += 1;
            for (ji, gplan) in plan {
                let j = &mut jobs[ji];
                let q = &qs[j.q];
                tokens[j.state] += gplan.iter().flat_map(|(_, ids)| ids).map(|&r| rows[r].ids.len()).sum::<usize>();
                if j.rounds == 0 && text_embedder.is_none() {
                    if let Some(ex) = &exp[j.q] {
                        // non-shortlisted: the first row is exactly the full-question row
                        let r = embed_row.get(&ji).copied().unwrap_or(gplan[0].1[0]);
                        tokens[j.state] += if embed_row.contains_key(&ji) { rows[r].ids.len() } else { 0 };
                        if let Some(v) = &scored[r].pooled {
                            j.p_mem = Some(knn(v, ex, q.n_options(), opts.experience_k, opts.experience_temperature));
                            j.votes = Some(votes_excluding(v, ex, q.n_options(), opts.experience_k, None));
                        }
                    }
                }
                j.rounds += 1;
                let mut group_p: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
                for (grp, ids) in &gplan {
                    let k = grp.len();
                    let mut p = vec![0.0; k];
                    for &r in ids {
                        let sp = softmax_t(&scored[r].logits, self.model.temperature(q.t, k));
                        // markers[m] holds option order[m] of this group
                        for (m, &opt) in rows[r].order.iter().enumerate() {
                            p[opt] += sp[m] / ids.len() as f64;
                        }
                        j.act = scored[r].act_probability;
                    }
                    group_p.push((grp.clone(), p));
                }
                if group_p.len() == 1 {
                    let (grp, mut p) = group_p.pop().unwrap();
                    if let Some(first) = j.pending_twin.take() {
                        // adaptive ensemble completed: average the two single-order passes
                        for (x, f) in p.iter_mut().zip(first) {
                            *x = (*x + f) / 2.0;
                        }
                        j.twinned = true;
                    } else if opts.permute && opts.permute_below_confidence > 0.0 && grp.len() >= 2 {
                        let conf = if q.t == QType::Noul { p[0].max(p[1]) } else { confidence_from_probs(&p) };
                        if conf < opts.permute_below_confidence {
                            j.pending_twin = Some(p);
                            continue; // stays alive for one more round with its reversed row
                        }
                    }
                    let mut full = vec![0.0; q.n_options()];
                    for (i, &o) in grp.iter().enumerate() {
                        full[o] = p[i];
                    }
                    let laya_margin = {
                        let mut v: Vec<f64> = grp.iter().map(|&o| full[o]).collect();
                        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        v.first().copied().unwrap_or(0.0) - v.get(1).copied().unwrap_or(0.0)
                    };
                    if let Some((pm, _)) = j.p_mem.as_ref().filter(|_| laya_margin < opts.experience_below_margin) {
                        // Laya disposes: log-linear pool over the options still alive
                        let mut z = 0.0;
                        for &o in &grp {
                            full[o] = ((full[o].max(1e-12)).ln() + exp_w[j.q] * (pm[o] + 1e-3).ln()).exp();
                            z += full[o];
                        }
                        for &o in &grp {
                            full[o] /= z.max(1e-300);
                        }
                    }
                    let mut grp = grp;
                    let fire = opts.experience_override_agree > 0 && exp_loo[j.q] >= opts.experience_override_min_trust
                        && j.votes.as_ref().is_some_and(|v| v.iter().copied().max().unwrap_or(0) >= opts.experience_override_agree);
                    if fire {
                        // the neighbours agree and the memory is reliable: their vote decides
                        let v = j.votes.as_ref().unwrap();
                        let tot = v.iter().sum::<usize>().max(1) as f64;
                        for (x, &c) in full.iter_mut().zip(v) {
                            *x = c as f64 / tot;
                        }
                        for (o, &c) in v.iter().enumerate() {
                            if c > 0 && !grp.contains(&o) {
                                grp.push(o);
                            }
                        }
                        grp.sort_unstable();
                        j.overridden = true;
                    }
                    j.alive = grp;
                    j.final_p = Some(full);
                } else {
                    let f = opts.finalists_per_group.max(1);
                    let mut next = Vec::new();
                    for (grp, p) in group_p {
                        let mut idx: Vec<usize> = (0..grp.len()).collect();
                        idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
                        next.extend(idx.into_iter().take(f).map(|i| grp[i]));
                    }
                    next.extend(j.emb_top.iter().copied()); // union mode: embedding shortlist joins
                    if let Some((pm, _)) = j.p_mem.as_ref().filter(|_| exp_w[j.q] > 0.0) {
                        // memory proposes (only when trusted): its top labels join the next round
                        let mut idx: Vec<usize> = (0..pm.len()).collect();
                        idx.sort_by(|&a, &b| pm[b].partial_cmp(&pm[a]).unwrap());
                        next.extend(idx.into_iter().take(opts.experience_candidates).filter(|&i| pm[i] > 0.0));
                    }
                    next.sort_unstable(); // keep caller order among finalists
                    next.dedup();
                    j.alive = next;
                }
            }
        }

        for j in jobs {
            let q: &Question = &qs[j.q];
            let p = j.final_p.unwrap();
            let mut a = answer_json_with(q, &p, j.act, opts.round_probabilities);
            // One confidence per answer, with Laya's per-type definition (entropy for choice/score,
            // max(p, 1-p) for noul). Only a shortlisted choice differs: its entropy confidence is
            // over the final round's options (eliminated options have p = 0 and are not counted).
            if j.shortlisted {
                let fin: Vec<f64> = j.alive.iter().map(|&i| p[i]).collect();
                let c = confidence_from_probs(&fin);
                a["confidence"] = json!(if opts.round_probabilities { super::round4(c) } else { c });
            }
            let conf = a["confidence"].as_f64().unwrap_or(0.0);
            let mut meta = json!({
                "strategy": if j.shortlisted { "tournament" } else { "direct" },
                "permuted": (opts.permute && (opts.permute_below_confidence == 0.0 || j.twinned))
                    || (j.shortlisted && opts.permute_shortlisted_final && opts.permute_below_confidence == 0.0),
                "rounds": j.rounds,
                "handoff": opts.min_confidence > 0.0 && conf < opts.min_confidence,
                "out_of_distribution": false,
                "cached": false,
            });
            if let Some((pm, k)) = &j.p_mem {
                let best = pm.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
                meta["experience"] = json!({"neighbors": k, "memory_top": best, "memory_top_p": super::round4(pm[best]), "effective_weight": super::round4(exp_w[j.q]),
                    "override": j.overridden});
            }
            if j.shortlisted {
                let labels = q.choice_labels();
                meta["shortlist"] = json!(j.alive.iter().map(|&i| labels[i].clone()).collect::<Vec<_>>());
            }
            a["bodi"] = meta;
            if opts.cache {
                let key = format!("{}\u{1}{}\u{1}{opt_key}", sequence::serialize_state(&states[j.state]), qdefs[j.q]);
                self.cache.lock().unwrap().put(key, a.clone());
            }
            out[j.state].insert(q.id.clone(), a);
        }

        {
            let mut s = self.stats.lock().unwrap();
            s.decisions += stats.decisions;
            s.cache_hits += stats.cache_hits;
            s.rows_scored += stats.rows_scored;
            s.forward_rounds += stats.forward_rounds;
            s.script_gated += stats.script_gated;
        }
        // restore the caller's question order per state
        Ok(out
            .into_iter()
            .zip(tokens)
            .map(|(mut m, t)| {
                let mut answers = Map::new();
                for q in &qs {
                    if let Some(v) = m.remove(&q.id) {
                        answers.insert(q.id.clone(), v);
                    }
                }
                json!({"model": "bodi", "answers": answers, "usage": {"input_tokens": t, "output_tokens": 0}})
            })
            .collect())
    }
}
