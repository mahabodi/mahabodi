//! Rust vs Laya (PyTorch) parity on golden cases from research/golden_parity.py.
//!
//! Needs the exported model (research/export_onnx.py -> models/laya-v2) and libonnxruntime,
//! so these are #[ignore]d by default. Run explicitly:
//!     BODI_LAYA_DIR=$PWD/models/laya-v2 cargo test -p mahabodi-core --test laya_parity -- --ignored
#![cfg(feature = "laya")]

use mahabodi_core::system1::decide::{DecideOptions, System1};
use mahabodi_core::system1::sequence::parse_questions;
use mahabodi_core::system1::{LayaModel, LoadOptions};
use serde_json::Value;

fn model_dir() -> String {
    std::env::var("BODI_LAYA_DIR").unwrap_or_else(|_| format!("{}/../../models/laya-v2", env!("CARGO_MANIFEST_DIR")))
}

fn golden() -> Vec<Value> {
    // BODI_LAYA_GOLDEN selects another checkpoint's golden file (e.g. laya_parity_multilingual.json)
    let f = std::env::var("BODI_LAYA_GOLDEN").unwrap_or_else(|_| "laya_parity.json".into());
    let p = format!("{}/tests/fixtures/{f}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn load() -> LayaModel {
    LayaModel::load(model_dir(), &LoadOptions::default()).expect("load the Laya model (BODI_LAYA_DIR or models/laya-v2)")
}

#[test]
#[ignore = "needs models/laya-v2 (or BODI_LAYA_DIR) + libonnxruntime"]
fn token_ids_and_markers_match_laya_exactly() {
    let m = load();
    let mut rows = 0;
    for case in golden() {
        let qs = parse_questions(&case["questions"]).unwrap();
        for (q, want) in qs.iter().zip(case["rows"].as_array().unwrap()) {
            let e = m.encode(&case["state"], q, None);
            let ids: Vec<u64> = want["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let markers: Vec<u64> = want["markers"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            let got: Vec<u64> = e.ids.iter().map(|&x| x as u64).collect();
            assert_eq!(got, ids, "ids differ for question {}", q.id);
            assert_eq!(e.markers.iter().map(|&x| x as u64).collect::<Vec<_>>(), markers, "markers differ for {}", q.id);
            rows += 1;
        }
    }
    assert!(rows >= 8, "{rows}");
}

fn assert_answers_close(got: &Value, want: &Value, tol: f64, ctx: &str) {
    for (qid, w) in want.as_object().unwrap() {
        let g = &got[qid];
        assert_eq!(g["type"], w["type"], "{ctx}/{qid}");
        for k in ["choice"] {
            if w.get(k).is_some() {
                assert_eq!(g[k], w[k], "{ctx}/{qid}: {k}");
            }
        }
        for k in ["score", "noul", "confidence"] {
            if let Some(wv) = w.get(k).and_then(Value::as_f64) {
                let gv = g[k].as_f64().unwrap();
                assert!((gv - wv).abs() <= tol, "{ctx}/{qid}: {k} {gv} vs laya {wv}");
            }
        }
        if let Some(wp) = w.get("probabilities").and_then(Value::as_object) {
            let gp = g["probabilities"].as_object().unwrap();
            assert_eq!(gp.keys().collect::<Vec<_>>(), wp.keys().collect::<Vec<_>>(), "{ctx}/{qid}: option order");
            for (k, v) in wp {
                let d = (gp[k].as_f64().unwrap() - v.as_f64().unwrap()).abs();
                assert!(d <= tol, "{ctx}/{qid}: p[{k}] differs by {d}");
            }
        }
    }
}

#[test]
#[ignore = "needs models/laya-v2 (or BODI_LAYA_DIR) + libonnxruntime"]
fn predictions_match_laya_pytorch() {
    let m = load();
    for (i, case) in golden().iter().enumerate() {
        let got = m.predict(&case["state"], &case["questions"]).unwrap();
        // 4-dp rounding on both sides + fp32 ORT vs torch: allow 2e-4
        assert_answers_close(&got["answers"], &case["laya"], 2e-4, &format!("case {i}"));
    }
}

#[test]
#[ignore = "needs models/laya-v2 (or BODI_LAYA_DIR) + libonnxruntime"]
fn laya_compatible_decide_equals_predict() {
    let s1 = System1::new(load());
    let opts = DecideOptions::laya_compatible();
    for (i, case) in golden().iter().enumerate() {
        let got = s1.decide(&case["state"], &case["questions"], &opts).unwrap();
        assert_answers_close(&got["answers"], &case["laya"], 2e-4, &format!("case {i}"));
    }
}

#[test]
#[ignore = "needs models/laya-v2 (or BODI_LAYA_DIR) + libonnxruntime"]
fn tournament_handles_77_options_and_gating_nulls_decisions() {
    let s1 = System1::new(load());
    let labels: Vec<String> = (0..77).map(|i| format!("intent number {i}")).collect();
    let q = serde_json::json!({"intent": {"type": "choice", "instructions": "Which intent?", "criteria": labels}});
    let a = s1.decide(&serde_json::json!("my card was declined"), &q, &DecideOptions::default()).unwrap();
    let ans = &a["answers"]["intent"];
    assert_eq!(ans["bodi"]["strategy"], "tournament");
    assert_eq!(ans["probabilities"].as_object().unwrap().len(), 77);
    assert!(ans["bodi"]["shortlist"].as_array().unwrap().len() <= 16);
    let khmer = s1.decide(&serde_json::json!("ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង"), &q, &DecideOptions::default()).unwrap();
    let k = &khmer["answers"]["intent"];
    assert!(k["choice"].is_null());
    assert_eq!(k["bodi"]["handoff"], true);
    // cache: an identical call is served from cache
    let again = s1.decide(&serde_json::json!("my card was declined"), &q, &DecideOptions::default()).unwrap();
    assert_eq!(again["answers"]["intent"]["bodi"]["cached"], true);
    assert_eq!(again["answers"]["intent"]["choice"], ans["choice"]);
}

#[test]
#[ignore = "needs models/minilm + libonnxruntime"]
fn embedder_matches_sentence_transformers() {
    let dir = format!("{}/../../models/minilm", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/golden.json")).unwrap()).unwrap();
    let e = bodi_embedder(&dir);
    let texts: Vec<String> = g["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    let got = e.embed(&texts).unwrap();
    for (v, w) in got.iter().zip(g["vectors"].as_array().unwrap()) {
        let cos: f64 = v.iter().zip(w.as_array().unwrap()).map(|(a, b)| *a as f64 * b.as_f64().unwrap()).sum();
        assert!(cos > 0.9999, "cosine {cos}");
    }
}

fn bodi_embedder(dir: &str) -> mahabodi_core::system1::embedder::Embedder {
    mahabodi_core::system1::embedder::Embedder::load(dir, None, 0).unwrap()
}
