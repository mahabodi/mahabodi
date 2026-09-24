//! Laya checkpoint executed natively through ONNX Runtime: no Python or torch at inference.
//!
//! Requires libonnxruntime 1.23+ (loaded dynamically). Resolution order: `LoadOptions::ort_dylib`,
//! `$ORT_DYLIB_PATH`, then - as a convenience only - the library bundled in a Python
//! `onnxruntime` install (found by running `$BODI_PYTHON`/`python3` once at load).
//! One ORT session per model behind a mutex: concurrent calls are serialized.
//!
//! Model directory layout (written by research/export_onnx.py):
//!   model.onnx (fused encoder + decision head), tokenizer.json, rl_agent_config.json

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::sequence::{self, Encoded, Encoder, QType, Question};
use super::{clamp_temperature, confidence_from_probs, round4, softmax_t, temp_bucket};
use crate::error::{BodiError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct LoadOptions {
    #[serde(default = "default_model_file")]
    pub model_file: String,
    /// ORT intra-op threads; 0 lets ONNX Runtime choose (physical cores).
    #[serde(default)]
    pub intra_threads: usize,
    /// Path to libonnxruntime (1.23+). Falls back to $ORT_DYLIB_PATH, then to the
    /// library shipped inside a Python `onnxruntime` install.
    #[serde(default)]
    pub ort_dylib: Option<PathBuf>,
    /// Rows per forward pass; rows are length-bucketed so each pass pads only to its own max.
    #[serde(default = "default_rows")]
    pub max_rows_per_pass: usize,
}

fn default_model_file() -> String {
    "model.onnx".into()
}
fn default_rows() -> usize {
    8
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { model_file: default_model_file(), intra_threads: 0, ort_dylib: None, max_rows_per_pass: default_rows() }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawCfg {
    #[serde(default)]
    encoder: String,
    #[serde(default = "d512")]
    max_len: usize,
    #[serde(default = "d192")]
    head_max_len: usize,
    #[serde(default)]
    temperature: Vec<f64>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f64>,
}
fn d512() -> usize {
    512
}
fn d192() -> usize {
    192
}

struct Tok {
    t: tokenizers::Tokenizer,
    mask_str: String,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
}

impl Encoder for Tok {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.t.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
    }
    fn cls(&self) -> u32 {
        self.cls
    }
    fn sep(&self) -> u32 {
        self.sep
    }
    fn mask(&self) -> u32 {
        self.mask
    }
    fn pad(&self) -> u32 {
        self.pad
    }
    fn mask_token(&self) -> &str {
        &self.mask_str
    }
}

/// One scored row: raw logits for its markers (in marker order) and P(act).
#[derive(Debug, Clone)]
pub struct Row {
    pub logits: Vec<f32>,
    pub act_probability: f64,
    /// [CLS] state after the decision head, when the export has a `pooled` output
    /// (research/export_onnx.py); used by experience memory. None for older exports.
    pub pooled: Option<Vec<f32>>,
}

pub struct LayaModel {
    session: Mutex<Session>,
    tok: Tok,
    pub max_len: usize,
    pub head_max_len: usize,
    temperature: [f64; 3],
    temperature_by_options: HashMap<String, f64>,
    /// ModernBERT (English BPE) checkpoints cannot read non-Latin scripts.
    pub english_only: bool,
    pub encoder: String,
    pub dir: PathBuf,
    max_rows: usize,
}

static ORT_INIT: OnceLock<std::result::Result<PathBuf, String>> = OnceLock::new();

fn find_ort_dylib(explicit: Option<&Path>) -> std::result::Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        return Ok(PathBuf::from(p));
    }
    let py = std::env::var("BODI_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = std::process::Command::new(&py)
        .args(["-c", "import onnxruntime,os;print(os.path.join(os.path.dirname(onnxruntime.__file__),'capi'))"])
        .output()
        .map_err(|e| format!("no ONNX Runtime library given and `{py}` failed: {e}"))?;
    let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    let lib = std::fs::read_dir(&dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            (n.starts_with("libonnxruntime") && (n.ends_with(".dylib") || n.contains(".so"))) || n == "onnxruntime.dll"
        })
        .ok_or_else(|| format!("no libonnxruntime in {}", dir.display()))?;
    Ok(lib)
}

pub(crate) fn init_ort(explicit: Option<&Path>) -> Result<()> {
    let r = ORT_INIT.get_or_init(|| {
        let lib = find_ort_dylib(explicit)?;
        ort::init_from(&lib).map_err(|e| format!("loading {}: {e}", lib.display()))?.commit();
        Ok(lib)
    });
    r.as_ref().map(|_| ()).map_err(|e| BodiError::Model(e.clone()))
}

impl LayaModel {
    pub fn load(dir: impl AsRef<Path>, opts: &LoadOptions) -> Result<LayaModel> {
        let dir = dir.as_ref().to_path_buf();
        init_ort(opts.ort_dylib.as_deref())?;
        let cfg: RawCfg = serde_json::from_str(&std::fs::read_to_string(dir.join("rl_agent_config.json"))?)?;
        let t = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| BodiError::Model(format!("tokenizer: {e}")))?;
        // Special tokens by the checkpoint's own names (ModernBERT: [CLS]/[SEP]/[MASK]/[PAD];
        // mmBERT/Gemma: <bos>/<eos>/<mask>/<pad>), as transformers' AutoTokenizer resolves them.
        let tcfg: serde_json::Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
        let name = |k: &str, d: &str| tcfg.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
        let (cls_s, sep_s, mask_s, pad_s) = (name("cls_token", "[CLS]"), name("sep_token", "[SEP]"), name("mask_token", "[MASK]"), name("pad_token", "[PAD]"));
        let id = |s: &str| t.token_to_id(s).ok_or_else(|| BodiError::Model(format!("tokenizer has no {s}")));
        let tok = Tok { cls: id(&cls_s)?, sep: id(&sep_s)?, mask: id(&mask_s)?, pad: id(&pad_s)?, mask_str: mask_s, t };
        let model_err = |e: ort::Error| BodiError::Model(e.to_string());
        let mut b = Session::builder().map_err(model_err)?;
        b = b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| BodiError::Model(e.to_string()))?;
        if opts.intra_threads > 0 {
            b = b.with_intra_threads(opts.intra_threads).map_err(|e| BodiError::Model(e.to_string()))?;
        }
        let session = b.commit_from_file(dir.join(&opts.model_file)).map_err(model_err)?;
        let temp = |i: usize| clamp_temperature(cfg.temperature.get(i).copied().unwrap_or(1.0));
        Ok(LayaModel {
            session: Mutex::new(session),
            tok,
            max_len: cfg.max_len,
            head_max_len: cfg.head_max_len,
            temperature: [temp(0), temp(1), temp(2)],
            temperature_by_options: cfg.temperature_by_options.iter().map(|(k, v)| (k.clone(), clamp_temperature(*v))).collect(),
            english_only: cfg.encoder.to_lowercase().contains("modernbert"),
            encoder: cfg.encoder,
            dir,
            max_rows: opts.max_rows_per_pass.max(1),
        })
    }

    pub fn encode(&self, state: &Value, q: &Question, order: Option<&[usize]>) -> Encoded {
        // Laya truncates conversation lists from the left so the newest turn survives.
        sequence::build_sequence(&self.tok, state, q, self.max_len, self.head_max_len, order, state.is_array())
    }

    pub fn temperature(&self, qt: QType, k: usize) -> f64 {
        self.temperature_by_options.get(&temp_bucket(qt, k)).copied().unwrap_or(self.temperature[qt as usize])
    }

    /// Score rows. Rows are sorted by length and run `max_rows` at a time, each pass padded
    /// only to its own longest row (Laya pads a whole call to its global max).
    pub fn run(&self, rows: &[&Encoded]) -> Result<Vec<Row>> {
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let mut out: Vec<Option<Row>> = vec![None; rows.len()];
        for chunk in order.chunks(self.max_rows) {
            let b = chunk.len();
            let l = chunk.iter().map(|&i| rows[i].ids.len()).max().unwrap_or(1);
            let k = chunk.iter().map(|&i| rows[i].markers.len()).max().unwrap_or(1).max(1);
            let mut ids = vec![self.tok.pad as i64; b * l];
            let mut att = vec![0i64; b * l];
            let mut mpos = vec![0i64; b * k];
            let mut mmask = vec![false; b * k];
            let mut qt = vec![0i64; b];
            for (r, &i) in chunk.iter().enumerate() {
                let e = rows[i];
                for (j, &t) in e.ids.iter().enumerate() {
                    ids[r * l + j] = t as i64;
                    att[r * l + j] = 1;
                }
                for (j, &m) in e.markers.iter().enumerate() {
                    mpos[r * k + j] = m as i64;
                    mmask[r * k + j] = true;
                }
                qt[r] = e.qtype as i64;
            }
            let t = |e: ort::Error| BodiError::Model(e.to_string());
            let inputs = ort::inputs![
                "input_ids" => Tensor::from_array(([b, l], ids)).map_err(t)?,
                "attention_mask" => Tensor::from_array(([b, l], att)).map_err(t)?,
                "marker_pos" => Tensor::from_array(([b, k], mpos)).map_err(t)?,
                "marker_mask" => Tensor::from_array(([b, k], mmask)).map_err(t)?,
                "qtype" => Tensor::from_array(([b], qt)).map_err(t)?,
            ];
            let mut sess = self.session.lock().map_err(|_| BodiError::Model("session lock poisoned".into()))?;
            let res = sess.run(inputs).map_err(t)?;
            let (_, logits) = res["logits"].try_extract_tensor::<f32>().map_err(t)?;
            let (ashape, act) = res["act_logits"].try_extract_tensor::<f32>().map_err(t)?;
            let na = ashape[1] as usize;
            let pooled = match res.get("pooled") {
                Some(v) => {
                    let (ps, pd) = v.try_extract_tensor::<f32>().map_err(t)?;
                    Some((ps[1] as usize, pd.to_vec()))
                }
                None => None,
            };
            for (r, &i) in chunk.iter().enumerate() {
                let kr = rows[i].markers.len();
                let a = &act[r * na..(r + 1) * na];
                let pa = softmax_t(a, 1.0);
                let pv = pooled.as_ref().map(|(d, data)| data[r * d..(r + 1) * d].to_vec());
                out[i] = Some(Row { logits: logits[r * k..r * k + kr].to_vec(), act_probability: pa[0], pooled: pv });
            }
        }
        Ok(out.into_iter().map(|r| r.unwrap()).collect())
    }

    pub fn n_tokens(rows: &[&Encoded]) -> usize {
        rows.iter().map(|e| e.ids.len()).sum()
    }

    /// Laya-compatible `predict`: same inputs, same output schema, same maths.
    pub fn predict(&self, state: &Value, questions: &Value) -> Result<Value> {
        let qs = sequence::parse_questions(questions)?;
        if qs.is_empty() {
            return Ok(json!({"model": "bodi-laya-onnx", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}));
        }
        let encs: Vec<Encoded> = qs.iter().map(|q| self.encode(state, q, None)).collect();
        for (q, e) in qs.iter().zip(&encs) {
            if e.markers.len() != q.n_options() {
                return Err(BodiError::Invalid(format!(
                    "question {:?}: only {} of {} option markers fit in max_len={} (head_max_len={})",
                    q.id, e.markers.len(), q.n_options(), self.max_len, self.head_max_len)));
            }
        }
        let refs: Vec<&Encoded> = encs.iter().collect();
        let rows = self.run(&refs)?;
        let mut answers = Map::new();
        for (q, row) in qs.iter().zip(&rows) {
            let p = softmax_t(&row.logits, self.temperature(q.t, row.logits.len()));
            answers.insert(q.id.clone(), answer_json(q, &p, row.act_probability));
        }
        Ok(json!({"model": "bodi-laya-onnx", "answers": answers, "usage": {"input_tokens": Self::n_tokens(&refs), "output_tokens": 0}}))
    }
}

/// laya/agent.py `_decode_answers` for one question given final probabilities in option order.
pub fn answer_json(q: &Question, p: &[f64], act_probability: f64) -> Value {
    answer_json_with(q, p, act_probability, true)
}

/// `round`: Laya's schema rounds every number to 4 decimals. With `round = false` the exact
/// probabilities are returned (needed for NLL/Brier: a rounded 0.0 has log-probability -inf).
pub fn answer_json_with(q: &Question, p: &[f64], act_probability: f64, round: bool) -> Value {
    let round4 = |x: f64| if round { round4(x) } else { x };
    let conf = round4(confidence_from_probs(p));
    let action = json!({"act_probability": round4(act_probability)});
    match q.t {
        QType::Choice => {
            let keys = q.choice_labels();
            let best = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
            let mut probs = Map::new();
            for (k, v) in keys.iter().zip(p) {
                probs.insert(k.clone(), json!(round4(*v)));
            }
            json!({"type": "choice", "choice": keys[best], "probabilities": probs, "confidence": conf, "action": action})
        }
        QType::Score => {
            let exp: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
            let levels = match &q.crit {
                sequence::Criteria::Score(l) => l.clone(),
                _ => Vec::new(),
            };
            let legend: Map<String, Value> = levels.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect();
            let probs: Map<String, Value> = p.iter().enumerate().map(|(i, v)| (i.to_string(), json!(round4(*v)))).collect();
            json!({"type": "score", "score": round4(exp), "legend": legend, "probabilities": probs, "confidence": conf, "action": action})
        }
        QType::Noul => {
            let pt = p.get(1).copied().unwrap_or(0.5);
            json!({"type": "noul", "noul": round4(pt), "confidence": round4(pt.max(1.0 - pt)), "action": action})
        }
    }
}
