//! System 1: fast, typed, calibrated decisions from a Laya checkpoint.
//!
//! `sequence` and `script` are pure Rust and always compiled. The ONNX runtime
//! (`LayaModel`) and Bodi's decision strategies (`decide`) need the `laya` feature.

pub mod script;
pub mod sequence;

#[cfg(feature = "laya")]
mod runtime;
#[cfg(feature = "laya")]
pub mod decide;
#[cfg(feature = "laya")]
pub mod embedder;
#[cfg(feature = "laya")]
pub use runtime::{LayaModel, LoadOptions, Row};

/// laya/common.py `temp_bucket`.
pub fn temp_bucket(qtype: sequence::QType, k: usize) -> String {
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    format!("{}:{}", qtype.name(), size)
}

/// laya/common.py `clamp_temperature` (TEMP_MIN 0.5, TEMP_MAX 5.0).
pub fn clamp_temperature(t: f64) -> f64 {
    if !t.is_finite() {
        return 1.0;
    }
    t.clamp(0.5, 5.0)
}

/// laya/common.py `confidence_from_probs`: 1 - H(p)/log(k).
pub fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = -p.iter().map(|&x| x * x.clamp(1e-12, 1.0).ln()).sum::<f64>();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

pub fn softmax_t(logits: &[f32], t: f64) -> Vec<f64> {
    let z: Vec<f64> = logits.iter().map(|&x| x as f64 / t).collect();
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

pub fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}
