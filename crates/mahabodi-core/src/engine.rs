//! `Bodi`: the facade every binding wraps. Thread-safe (`Send + Sync`); memory reads run
//! concurrently, ingestion takes a write lock, model inference is serialized per session.
//!
//! `call(method, args)` is the JSON dispatcher the C ABI and the language bindings use, so
//! every language exposes exactly the same operations with the same argument names.

use std::sync::RwLock;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::density::{self, DensityPolicy};
use crate::error::{BodiError, Result};
use crate::ingest::Format;
use crate::memory::Memory;
use crate::query;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BodiConfig {
    pub density: DensityPolicy,
    /// Community detection: "deterministic" (default) or "fast_memory" (fastmemory's own).
    pub cluster_engine: crate::graph::ClusterEngine,
    /// Run the density guard after every ingest (default true).
    pub auto_density: bool,
    /// Load a Laya ONNX model directory at construction.
    pub laya_dir: Option<String>,
    /// Load a dense text embedder (research/export_embedder.py) at construction. Enables hybrid
    /// (BM25 cascade + dense, RRF-fused) memory retrieval and text-space experience memory.
    pub embedder_dir: Option<String>,
    /// Hybrid retrieval: a dense cosine at/above this is a match on its own (tuned on validation).
    pub dense_min_similarity: f32,
    /// Hybrid retrieval: typo-corrected matches must also be dense-similar (opt-in caution).
    pub fuzzy_requires_dense: bool,
    #[cfg(feature = "laya")]
    pub laya: Option<crate::system1::LoadOptions>,
}

impl Default for BodiConfig {
    fn default() -> Self {
        BodiConfig {
            density: DensityPolicy::default(),
            cluster_engine: Default::default(),
            auto_density: true,
            laya_dir: None,
            embedder_dir: None,
            // tuned on SQuAD train (research/results/tune_hybrid.json)
            dense_min_similarity: 0.35,
            fuzzy_requires_dense: false,
            #[cfg(feature = "laya")]
            laya: None,
        }
    }
}

pub struct Bodi {
    memory: RwLock<Memory>,
    config: BodiConfig,
    #[cfg(feature = "laya")]
    s1: RwLock<Option<std::sync::Arc<crate::system1::decide::System1>>>,
    #[cfg(feature = "laya")]
    embedder: RwLock<Option<std::sync::Arc<crate::system1::embedder::Embedder>>>,
}

fn arg<'a>(a: &'a Value, k: &str) -> Result<&'a Value> {
    a.get(k).ok_or_else(|| BodiError::Invalid(format!("missing argument '{k}'")))
}

fn arg_str<'a>(a: &'a Value, k: &str) -> Result<&'a str> {
    arg(a, k)?.as_str().ok_or_else(|| BodiError::Invalid(format!("argument '{k}' must be a string")))
}

fn opt_usize(a: &Value, k: &str, d: usize) -> usize {
    a.get(k).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d)
}

impl Bodi {
    pub fn new(config: BodiConfig) -> Result<Bodi> {
        let b = Bodi {
            memory: RwLock::new(Memory::with_engine(config.cluster_engine)),
            #[cfg(feature = "laya")]
            s1: RwLock::new(None),
            #[cfg(feature = "laya")]
            embedder: RwLock::new(None),
            config,
        };
        #[cfg(feature = "laya")]
        if let Some(dir) = b.config.laya_dir.clone() {
            let opts = b.config.laya.clone().unwrap_or_default();
            b.load_laya(&dir, &opts)?;
        }
        #[cfg(feature = "laya")]
        if let Some(dir) = b.config.embedder_dir.clone() {
            b.load_embedder(&dir, &b.config.laya.clone().unwrap_or_default())?;
        }
        #[cfg(not(feature = "laya"))]
        if b.config.laya_dir.is_some() {
            return Err(BodiError::Model("this build has no `laya` feature".into()));
        }
        Ok(b)
    }

    pub fn from_json(config: &str) -> Result<Bodi> {
        let c: BodiConfig = if config.trim().is_empty() { BodiConfig::default() } else { serde_json::from_str(config)? };
        Bodi::new(c)
    }

    fn mem_r(&self) -> std::sync::RwLockReadGuard<'_, Memory> {
        self.memory.read().unwrap_or_else(|e| e.into_inner())
    }

    fn mem_w(&self) -> std::sync::RwLockWriteGuard<'_, Memory> {
        self.memory.write().unwrap_or_else(|e| e.into_inner())
    }

    pub fn ingest(&self, text: &str, format: Format, source: &str) -> Value {
        let mut m = self.mem_w();
        let rep = m.ingest(text, format, source);
        let dens = if self.config.auto_density { Some(density::ensure(&mut m, &self.config.density)) } else { None };
        json!({"ingest": rep, "density": dens.map(|d| json!({"passes": d.after.passes, "rounds": d.rounds, "concepts_added": d.concepts_added, "violations": d.after.violations}))})
    }

    /// Bulk ingest: one rebuild and one density pass for all documents.
    pub fn ingest_batch(&self, docs: &[(String, Format, String)]) -> Value {
        let mut m = self.mem_w();
        let refs: Vec<(&str, Format, &str)> = docs.iter().map(|(t, f, s)| (t.as_str(), *f, s.as_str())).collect();
        let rep = m.ingest_many(&refs);
        let dens = if self.config.auto_density { Some(density::ensure(&mut m, &self.config.density)) } else { None };
        json!({"ingest": rep, "density": dens.map(|d| json!({"passes": d.after.passes, "rounds": d.rounds, "concepts_added": d.concepts_added, "violations": d.after.violations}))})
    }

    pub fn density(&self) -> Value {
        json!(density::measure(&self.mem_r(), &self.config.density))
    }

    pub fn ensure_density(&self) -> Value {
        json!(density::ensure(&mut self.mem_w(), &self.config.density))
    }

    /// Hybrid when an embedder is loaded: the query is embedded outside the memory lock.
    fn dense_query(&self, q: &str) -> Option<Vec<f32>> {
        let e = self.mem_r().embedder()?;
        e.embed_texts(&[q.to_string()]).ok()?.pop()
    }

    pub fn query(&self, q: &str, k: usize) -> Value {
        let qv = self.dense_query(q);
        let m = self.mem_r();
        let d = qv.map(|v| query::Dense { vecs: m.dense(), query: v, min_similarity: self.config.dense_min_similarity, fuzzy_requires_dense: self.config.fuzzy_requires_dense });
        json!(query::query_with(m.graph(), m.index(), q, k, d.as_ref()))
    }

    pub fn traverse(&self, start: &str, hops: usize, limit: usize) -> Value {
        let m = self.mem_r();
        json!(query::traverse(m.graph(), m.index(), start, hops, limit))
    }

    pub fn context(&self, q: &str, k: usize, max_chars: usize) -> Value {
        let qv = self.dense_query(q);
        let m = self.mem_r();
        let d = qv.map(|v| query::Dense { vecs: m.dense(), query: v, min_similarity: self.config.dense_min_similarity, fuzzy_requires_dense: self.config.fuzzy_requires_dense });
        let (c, r) = query::context_with(m.graph(), m.index(), q, k, max_chars, d.as_ref());
        json!({"context": c, "matched": r.matched, "handoff": r.handoff, "stage": r.stage, "confidence": r.confidence})
    }

    /// What plain fastmemory's own search returns on the same memory (for comparison).
    pub fn fastmemory_search(&self, q: &str) -> Value {
        query::fastmemory_search(&self.mem_r().fastmemory_json(), q)
    }

    pub fn snapshot(&self) -> Value {
        let m = self.mem_r();
        json!({"version": 1, "atfs": m.atfs, "links": m.links, "texts": m.texts, "concepts": m.concepts})
    }

    pub fn restore(&self, snap: &Value) -> Result<Value> {
        let mut m = self.mem_w();
        let mut fresh = Memory::with_engine(self.config.cluster_engine);
        fresh.atfs = serde_json::from_value(snap.get("atfs").cloned().unwrap_or(json!([])))?;
        fresh.links = serde_json::from_value(snap.get("links").cloned().unwrap_or(json!([])))?;
        fresh.texts = serde_json::from_value(snap.get("texts").cloned().unwrap_or(json!({})))?;
        fresh.concepts = serde_json::from_value(snap.get("concepts").cloned().unwrap_or(json!([])))?;
        fresh.set_embedder(m.embedder()); // keeps hybrid retrieval across restore
        fresh.rebuild();
        *m = fresh;
        Ok(json!({"atfs": m.atfs.len(), "nodes": m.graph().nodes.len()}))
    }

    pub fn clear(&self) {
        self.mem_w().clear();
    }

    pub fn stats(&self) -> Value {
        let m = self.mem_r();
        let g = m.graph();
        #[allow(unused_mut)]
        let mut v = json!({"atfs": m.atfs().len(), "nodes": g.nodes.len(), "edges": g.edge_count, "blocks": g.blocks.len(),
                           "vocabulary": m.index().vocab_size(), "laya_loaded": false});
        #[cfg(feature = "laya")]
        if let Some(s1) = self.s1.read().unwrap().as_ref() {
            v["laya_loaded"] = json!(true);
            v["system1"] = json!(s1.stats());
            v["experience_cases"] = json!(s1.experience_size());
            v["laya"] = json!({"dir": s1.model.dir, "encoder": s1.model.encoder, "english_only": s1.model.english_only});
        }
        #[cfg(feature = "laya")]
        if let Some(e) = self.embedder.read().unwrap().as_ref() {
            v["embedder"] = json!({"dir": e.dir, "max_len": e.max_len});
        }
        v
    }

    /// JSON dispatcher shared by all bindings.
    pub fn call(&self, method: &str, args: &Value) -> Result<Value> {
        let a = args;
        Ok(match method {
            "ingest" => {
                let format: Format = match a.get("format") {
                    Some(f) => serde_json::from_value(f.clone())?,
                    None => Format::Auto,
                };
                self.ingest(arg_str(a, "text")?, format, a.get("source").and_then(Value::as_str).unwrap_or("doc"))
            }
            "ingest_batch" => {
                let docs = arg(a, "docs")?.as_array().ok_or_else(|| BodiError::Invalid("'docs' must be an array".into()))?;
                let mut v = Vec::with_capacity(docs.len());
                for d in docs {
                    let format: Format = match d.get("format") {
                        Some(f) => serde_json::from_value(f.clone())?,
                        None => Format::Auto,
                    };
                    v.push((arg_str(d, "text")?.to_string(), format, d.get("source").and_then(Value::as_str).unwrap_or("doc").to_string()));
                }
                self.ingest_batch(&v)
            }
            "density" => self.density(),
            "ensure_density" => self.ensure_density(),
            "query" => self.query(arg_str(a, "q")?, opt_usize(a, "k", 5)),
            "traverse" => self.traverse(arg_str(a, "start")?, opt_usize(a, "hops", 2), opt_usize(a, "limit", 50)),
            "context" => self.context(arg_str(a, "q")?, opt_usize(a, "k", 5), opt_usize(a, "max_chars", 2000)),
            "fastmemory_search" => self.fastmemory_search(arg_str(a, "q")?),
            "snapshot" => self.snapshot(),
            "restore" => self.restore(arg(a, "snapshot")?)?,
            "clear" => {
                self.clear();
                json!({"ok": true})
            }
            "stats" => self.stats(),
            #[cfg(feature = "laya")]
            "load_laya" => {
                let opts: crate::system1::LoadOptions = match a.get("options") {
                    Some(o) => serde_json::from_value(o.clone())?,
                    None => Default::default(),
                };
                self.load_laya(arg_str(a, "dir")?, &opts)?;
                json!({"ok": true})
            }
            #[cfg(feature = "laya")]
            "decide" => self.decide(arg(a, "state")?, arg(a, "questions")?, &decide_opts(a)?)?,
            #[cfg(feature = "laya")]
            "learn" => {
                let states = arg(a, "states")?.as_array().ok_or_else(|| BodiError::Invalid("'states' must be an array".into()))?;
                let labels = arg(a, "labels")?.as_array().ok_or_else(|| BodiError::Invalid("'labels' must be an array".into()))?;
                let s1 = self.system1()?;
                let mut r = s1.learn(states, arg(a, "questions")?, labels)?;
                let cal = opt_usize(a, "calibrate", 0);
                if cal > 0 {
                    r["calibration"] = s1.calibrate(states, arg(a, "questions")?, labels, cal)?;
                }
                r
            }
            #[cfg(feature = "laya")]
            "load_embedder" => {
                let opts: crate::system1::LoadOptions = match a.get("options") {
                    Some(o) => serde_json::from_value(o.clone())?,
                    None => Default::default(),
                };
                self.load_embedder(arg_str(a, "dir")?, &opts)?;
                json!({"ok": true})
            }
            #[cfg(feature = "laya")]
            "embed_text" => {
                let texts: Vec<String> = serde_json::from_value(arg(a, "texts")?.clone())?;
                let e = self.embedder.read().unwrap().clone().ok_or_else(|| BodiError::Model("no embedder loaded: call load_embedder".into()))?;
                json!(e.embed(&texts)?)
            }
            #[cfg(feature = "laya")]
            "embed" => {
                let states = arg(a, "states")?.as_array().ok_or_else(|| BodiError::Invalid("'states' must be an array".into()))?;
                json!(self.system1()?.embed(states, arg(a, "questions")?)?)
            }
            #[cfg(feature = "laya")]
            "decide_defaults" => serde_json::to_value(crate::system1::decide::DecideOptions::default())?,
            #[cfg(feature = "laya")]
            "forget" => {
                self.system1()?.forget();
                json!({"ok": true})
            }
            #[cfg(feature = "laya")]
            "predict" => self.system1()?.model.predict(arg(a, "state")?, arg(a, "questions")?)?,
            #[cfg(feature = "laya")]
            "decide_batch" => {
                let states = arg(a, "states")?.as_array().ok_or_else(|| BodiError::Invalid("'states' must be an array".into()))?;
                json!(self.system1()?.decide_batch(states, arg(a, "questions")?, &decide_opts(a)?)?)
            }
            #[cfg(feature = "laya")]
            "decide_with_memory" => self.decide_with_memory(
                arg(a, "state")?,
                arg(a, "questions")?,
                &decide_opts(a)?,
                a.get("query").and_then(Value::as_str),
                opt_usize(a, "k", 3),
                opt_usize(a, "max_chars", 1200),
                &match a.get("style") {
                    Some(Value::String(s)) if s == "passages" => GroundStyle::passages(),
                    Some(Value::String(s)) if s == "labelled" => GroundStyle::default(),
                    Some(Value::String(s)) => return Err(crate::error::BodiError::Invalid(format!("unknown grounding style {s:?} (passages | labelled | object)"))),
                    Some(v) if !v.is_null() => serde_json::from_value(v.clone())?,
                    _ => GroundStyle::default(),
                },
            )?,
            #[cfg(not(feature = "laya"))]
            "load_laya" | "decide" | "predict" | "decide_batch" | "decide_with_memory" | "learn" | "forget" | "embed" | "load_embedder" | "embed_text" | "decide_defaults" => {
                return Err(BodiError::Model("this build has no `laya` feature".into()))
            }
            other => return Err(BodiError::Invalid(format!("unknown method '{other}'"))),
        })
    }
}

/// Attach retrieved memory to a state without changing how Laya truncates it:
/// * conversation list: a `{"role": "system", "content": "memory: ..."}` turn goes just before
///   the newest turn, and the state stays a list, so left-truncation still keeps the newest turn;
/// * object: a trailing `memory` field; string: `{"input": ..., "memory": ...}`.
/// Empty context leaves the state untouched.
/// How retrieved memory is attached to a decision's state (`decide_with_memory`).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GroundStyle {
    /// Only the retrieved passages' text (no labels, no block/sibling lines).
    pub passages_only: bool,
    /// Passages to include (0 = all retrieved hits).
    pub top: usize,
    /// State key for the context (default "memory"; e.g. "passage" to match a passage question).
    pub key: String,
    /// Put the context before the state's own fields (e.g. passage, then question).
    pub first: bool,
}

impl Default for GroundStyle {
    fn default() -> Self {
        GroundStyle { passages_only: false, top: 0, key: "memory".into(), first: false }
    }
}

impl GroundStyle {
    /// `style: "passages"`: the top 3 retrieved passage texts under `passage`, before the state's
    /// fields. Chosen on a BoolQ dev sample disjoint from the test items (tune_grounding.json);
    /// BoolQ test 0.782 vs 0.656 for the default labelled format (bench_grounding_v2.json).
    /// Pair it with instructions that name `passage`.
    pub fn passages() -> Self {
        GroundStyle { passages_only: true, top: 3, key: "passage".into(), first: true }
    }
}

/// Attach context to an object state under `style.key`, first or last.
pub fn ground_state_styled(state: &Value, ctx: &str, style: &GroundStyle) -> Value {
    if ctx.is_empty() {
        return state.clone();
    }
    match state {
        Value::Object(o) => {
            let mut out = serde_json::Map::new();
            if style.first {
                out.insert(style.key.clone(), json!(ctx));
            }
            for (k, v) in o {
                if *k != style.key {
                    out.insert(k.clone(), v.clone());
                }
            }
            if !style.first {
                out.insert(style.key.clone(), json!(ctx));
            }
            Value::Object(out)
        }
        Value::Array(_) => ground_state(state, ctx),
        other => {
            let mut out = serde_json::Map::new();
            if style.first {
                out.insert(style.key.clone(), json!(ctx));
            }
            out.insert("input".into(), other.clone());
            if !style.first {
                out.insert(style.key.clone(), json!(ctx));
            }
            Value::Object(out)
        }
    }
}

pub fn ground_state(state: &Value, ctx: &str) -> Value {
    if ctx.is_empty() {
        return state.clone();
    }
    match state {
        Value::Array(turns) if !turns.is_empty() => {
            let mut t = turns.clone();
            let at = t.len() - 1;
            t.insert(at, json!({"role": "system", "content": format!("memory: {ctx}")}));
            Value::Array(t)
        }
        Value::Object(o) => {
            let mut o = o.clone();
            o.insert("memory".into(), json!(ctx));
            Value::Object(o)
        }
        other => json!({"input": other, "memory": ctx}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_grounding_places_context_first_under_key() {
        let st = GroundStyle { passages_only: true, top: 1, key: "passage".into(), first: true };
        let g = ground_state_styled(&json!({"question": "q?"}), "the text", &st);
        assert_eq!(serde_json::to_string(&g).unwrap(), r#"{"passage":"the text","question":"q?"}"#);
        assert_eq!(ground_state_styled(&json!({"a": 1}), "", &st), json!({"a": 1}));
    }

    #[test]
    fn grounding_keeps_list_states_lists_with_newest_turn_last() {
        let turns: Vec<Value> = (0..30).map(|i| json!({"role": "user", "content": format!("turn {i}")})).collect();
        let g = ground_state(&Value::Array(turns), "refunds take five days");
        let a = g.as_array().unwrap();
        assert_eq!(a.len(), 31);
        assert_eq!(a[30]["content"], "turn 29", "newest turn stays last (survives left truncation)");
        assert_eq!(a[29]["role"], "system");
        assert_eq!(ground_state(&json!("x"), ""), json!("x"));
        assert_eq!(ground_state(&json!({"a": 1}), "m"), json!({"a": 1, "memory": "m"}));
    }
}

#[cfg(feature = "laya")]
fn decide_opts(a: &Value) -> Result<crate::system1::decide::DecideOptions> {
    Ok(match a.get("options") {
        Some(o) if !o.is_null() => serde_json::from_value(o.clone())?,
        _ => Default::default(),
    })
}

#[cfg(feature = "laya")]
impl Bodi {
    pub fn load_laya(&self, dir: &str, opts: &crate::system1::LoadOptions) -> Result<()> {
        let m = crate::system1::LayaModel::load(dir, opts)?;
        let s1 = crate::system1::decide::System1::new(m);
        s1.set_embedder(self.embedder.read().unwrap().clone());
        *self.s1.write().unwrap() = Some(std::sync::Arc::new(s1));
        Ok(())
    }

    pub fn load_embedder(&self, dir: &str, opts: &crate::system1::LoadOptions) -> Result<()> {
        let e = std::sync::Arc::new(crate::system1::embedder::Embedder::load(dir, opts.ort_dylib.as_deref(), opts.intra_threads)?);
        if let Some(s1) = self.s1.read().unwrap().as_ref() {
            s1.set_embedder(Some(e.clone()));
        }
        self.mem_w().set_embedder(Some(e.clone() as std::sync::Arc<dyn crate::memory::TextEmbedder>));
        *self.embedder.write().unwrap() = Some(e);
        Ok(())
    }

    fn system1(&self) -> Result<std::sync::Arc<crate::system1::decide::System1>> {
        self.s1.read().unwrap().clone().ok_or(BodiError::NoModel)
    }

    pub fn decide(&self, state: &Value, questions: &Value, opts: &crate::system1::decide::DecideOptions) -> Result<Value> {
        self.system1()?.decide(state, questions, opts)
    }

    /// System 1 grounded in memory: the query (or the state's text) retrieves context, which is
    /// added to the state as `memory` - but only when retrieval did not hand off, so unrelated
    /// memory never leaks into a decision. The retrieval outcome is reported alongside.
    pub fn decide_with_memory(
        &self,
        state: &Value,
        questions: &Value,
        opts: &crate::system1::decide::DecideOptions,
        q: Option<&str>,
        k: usize,
        max_chars: usize,
        style: &GroundStyle,
    ) -> Result<Value> {
        let qtext = q.map(String::from).unwrap_or_else(|| crate::system1::sequence::serialize_state(state));
        let qv = self.dense_query(&qtext);
        let (ctx, r) = {
            let m = self.mem_r();
            let d = qv.map(|v| query::Dense { vecs: m.dense(), query: v, min_similarity: self.config.dense_min_similarity, fuzzy_requires_dense: self.config.fuzzy_requires_dense });
            let (full, r) = query::context_with(m.graph(), m.index(), &qtext, k, max_chars, d.as_ref());
            if style.passages_only && !r.handoff {
                let n = if style.top == 0 { r.hits.len() } else { style.top };
                let mut c = String::new();
                for h in r.hits.iter().filter(|h| !h.text.is_empty()).take(n) {
                    let t = h.text.as_str();
                    if !c.is_empty() && c.len() + t.len() + 2 > max_chars {
                        break;
                    }
                    if !c.is_empty() {
                        c.push_str("\n\n");
                    }
                    c.push_str(t);
                }
                (c, r)
            } else {
                (full, r)
            }
        };
        let grounded = if style.passages_only || style.key != "memory" || style.first {
            ground_state_styled(state, &ctx, style)
        } else {
            ground_state(state, &ctx)
        };
        let mut out = self.system1()?.decide(&grounded, questions, opts)?;
        out["memory"] = json!({"used": !ctx.is_empty(), "stage": r.stage, "matched": r.matched, "handoff": r.handoff,
                               "confidence": r.confidence, "hits": r.hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>()});
        Ok(out)
    }
}
