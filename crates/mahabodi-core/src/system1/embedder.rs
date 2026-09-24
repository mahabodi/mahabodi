//! Dense text embedder (sentence-transformers all-MiniLM-L6-v2, exported by
//! research/export_embedder.py) run through ONNX Runtime: token states -> attention-masked mean
//! -> L2 normalise, i.e. the sentence-transformers pipeline for this model. Parity with
//! SentenceTransformer.encode is tested against the exporter's golden vectors.

use std::path::Path;
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use crate::error::{BodiError, Result};

#[derive(Debug, Deserialize)]
struct Cfg {
    max_seq_length: usize,
}

pub struct Embedder {
    session: Mutex<Session>,
    tok: tokenizers::Tokenizer,
    pub max_len: usize,
    pub dir: std::path::PathBuf,
}

impl Embedder {
    pub fn load(dir: impl AsRef<Path>, ort_dylib: Option<&Path>, intra_threads: usize) -> Result<Embedder> {
        let dir = dir.as_ref().to_path_buf();
        super::runtime::init_ort(ort_dylib)?;
        let cfg: Cfg = serde_json::from_str(&std::fs::read_to_string(dir.join("embedder_config.json"))?)?;
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| BodiError::Model(format!("tokenizer: {e}")))?;
        // tokenizer.json ships truncation/padding at 128; sentence-transformers uses max_seq_length
        tok.with_truncation(Some(tokenizers::TruncationParams { max_length: cfg.max_seq_length, ..Default::default() }))
            .map_err(|e| BodiError::Model(format!("tokenizer: {e}")))?;
        tok.with_padding(None);
        let e = |x: ort::Error| BodiError::Model(x.to_string());
        let mut b = Session::builder().map_err(e)?;
        b = b.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|x| BodiError::Model(x.to_string()))?;
        if intra_threads > 0 {
            b = b.with_intra_threads(intra_threads).map_err(|x| BodiError::Model(x.to_string()))?;
        }
        let session = b.commit_from_file(dir.join("model.onnx")).map_err(e)?;
        Ok(Embedder { session: Mutex::new(session), tok, max_len: cfg.max_seq_length, dir })
    }

    /// Unit-norm embeddings, one per text (length-bucketed batches of 64).
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encs: Vec<tokenizers::Encoding> = texts
            .iter()
            .map(|t| self.tok.encode(t.as_str(), true).map_err(|e| BodiError::Model(format!("tokenize: {e}"))))
            .collect::<Result<_>>()?;
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| encs[i].get_ids().len());
        let mut out = vec![Vec::new(); texts.len()];
        for chunk in order.chunks(64) {
            let b = chunk.len();
            let l = chunk.iter().map(|&i| encs[i].get_ids().len()).max().unwrap_or(1).max(1);
            let (mut ids, mut att, mut tt) = (vec![0i64; b * l], vec![0i64; b * l], vec![0i64; b * l]);
            for (r, &i) in chunk.iter().enumerate() {
                let e = &encs[i];
                for (j, (&id, &ty)) in e.get_ids().iter().zip(e.get_type_ids()).enumerate() {
                    ids[r * l + j] = id as i64;
                    tt[r * l + j] = ty as i64;
                    att[r * l + j] = 1;
                }
            }
            let t = |e: ort::Error| BodiError::Model(e.to_string());
            let mask = att.clone();
            let inputs = ort::inputs![
                "input_ids" => Tensor::from_array(([b, l], ids)).map_err(t)?,
                "attention_mask" => Tensor::from_array(([b, l], att)).map_err(t)?,
                "token_type_ids" => Tensor::from_array(([b, l], tt)).map_err(t)?,
            ];
            let mut s = self.session.lock().map_err(|_| BodiError::Model("embedder lock poisoned".into()))?;
            let res = s.run(inputs).map_err(t)?;
            let (shape, h) = res["last_hidden_state"].try_extract_tensor::<f32>().map_err(t)?;
            let d = shape[2] as usize;
            for (r, &i) in chunk.iter().enumerate() {
                let mut v = vec![0f32; d];
                let mut n = 0f32;
                for j in 0..l {
                    if mask[r * l + j] == 1 {
                        n += 1.0;
                        let row = &h[(r * l + j) * d..(r * l + j + 1) * d];
                        for (a, x) in v.iter_mut().zip(row) {
                            *a += x;
                        }
                    }
                }
                let n = n.max(1e-9);
                v.iter_mut().for_each(|x| *x /= n);
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter_mut().for_each(|x| *x /= norm);
                out[i] = v;
            }
        }
        Ok(out)
    }
}

impl crate::memory::TextEmbedder for Embedder {
    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
}
