//! Exact port of Laya's question rendering and token-sequence construction
//! (laya/common.py `render_options`, `build_sequence`, `serialize_state`, and
//! laya/agent.py `_check_question`/`_to_internal`), so the ONNX model sees the
//! same ids it was trained on. Parity is tested against Python-built golden ids.

use serde_json::Value;

use crate::error::{BodiError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Criteria {
    /// (label, optional description), in the caller's key order.
    Choice(Vec<(String, Option<Value>)>),
    Score(Vec<Value>),
    Noul { false_: Option<Value>, true_: Option<Value>, labels: (String, String) },
}

#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub t: QType,
    pub ins: String,
    pub crit: Criteria,
}

impl Question {
    pub fn n_options(&self) -> usize {
        match &self.crit {
            Criteria::Choice(c) => c.len(),
            Criteria::Score(s) => s.len(),
            Criteria::Noul { .. } => 2,
        }
    }

    /// A copy restricted to the given choice options (in that order); used for shortlisting.
    pub fn with_choice_subset(&self, idx: &[usize]) -> Question {
        let mut q = self.clone();
        if let Criteria::Choice(c) = &self.crit {
            q.crit = Criteria::Choice(idx.iter().map(|&i| c[i].clone()).collect());
        }
        q
    }

    pub fn choice_labels(&self) -> Vec<String> {
        match &self.crit {
            Criteria::Choice(c) => c.iter().map(|(k, _)| k.clone()).collect(),
            _ => Vec::new(),
        }
    }
}

/// Python `json.dumps(v, ensure_ascii=False)` formatting (", " and ": " separators).
pub fn py_json(v: &Value) -> String {
    let mut s = String::new();
    write_py(v, &mut s, ", ", ": ");
    s
}

fn write_py(v: &Value, out: &mut String, item_sep: &str, key_sep: &str) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match n.as_f64().filter(|_| n.is_f64()) {
            Some(f) => out.push_str(&py_float(f)),
            None => out.push_str(&n.to_string()),
        },
        Value::String(s) => out.push_str(&serde_json::to_string(s).unwrap()),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_py(x, out, item_sep, key_sep);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(key_sep);
                write_py(x, out, item_sep, key_sep);
            }
            out.push('}');
        }
    }
}

/// Python `float.__repr__` (what json.dumps emits): shortest round-trip digits, positional
/// for 1e-4 <= |x| < 1e16 (always with a fractional part), otherwise `d[.ddd]e±XX`.
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{f:e}"); // shortest digits, e.g. "1.5e16", "1e-7"
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => py_json(v),
    }
}

/// Human-readable text of a state for semantic embedding: a string as-is; an object's values
/// (strings verbatim, anything else as JSON) joined by newlines; a conversation list's turns
/// (`content` when present) joined by newlines.
pub fn plain_text(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        Value::Object(o) => o.values().map(|v| match v {
            Value::String(s) => s.clone(),
            other => py_json(other),
        }).collect::<Vec<_>>().join("\n"),
        Value::Array(a) => a.iter().map(|t| match t.get("content") {
            Some(Value::String(c)) => c.clone(),
            _ => plain_text(t),
        }).collect::<Vec<_>>().join("\n"),
        other => py_json(other),
    }
}

/// laya/common.py `render_criterion`.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => py_json(v),
    }
}

fn is_blank(v: &Option<Value>) -> bool {
    match v {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    }
}

pub fn render_options(q: &Question) -> Vec<String> {
    match &q.crit {
        Criteria::Choice(c) => c
            .iter()
            .map(|(k, v)| if is_blank(v) { k.clone() } else { format!("{k}: {}", render_criterion(v.as_ref().unwrap())) })
            .collect(),
        Criteria::Score(levels) => levels.iter().enumerate().map(|(i, c)| format!("level {i}: {}", render_criterion(c))).collect(),
        Criteria::Noul { false_, true_, labels } => vec![
            format!("{}: {}", labels.0, if is_blank(false_) { "no, the statement does not hold".into() } else { render_criterion(false_.as_ref().unwrap()) }),
            format!("{}: {}", labels.1, if is_blank(true_) { "yes, the statement holds".into() } else { render_criterion(true_.as_ref().unwrap()) }),
        ],
    }
}

/// Validate + normalise one question definition (laya/agent.py `_check_question` + `_to_internal`).
pub fn parse_question(id: &str, def: &Value) -> Result<Question> {
    let bad = |m: String| BodiError::Invalid(format!("question {id:?}: {m}"));
    let o = def.as_object().ok_or_else(|| bad("definition must be an object".into()))?;
    let t = match o.get("type").and_then(Value::as_str) {
        Some("choice") => QType::Choice,
        Some("score") => QType::Score,
        Some("noul") => QType::Noul,
        other => return Err(bad(format!("unknown type {other:?}; use one of [choice, noul, score]"))),
    };
    let ins = match o.get("instructions") {
        None => return Err(bad("no 'instructions'; add the text the model should answer".into())),
        Some(Value::String(s)) => s.clone(),
        Some(v) => py_json(v),
    };
    if o.contains_key("labels") && t != QType::Noul {
        return Err(bad("'labels' is only supported for noul questions".into()));
    }
    let crit = o.get("criteria");
    let crit = match t {
        QType::Choice => match crit {
            Some(Value::Object(m)) if !m.is_empty() => Criteria::Choice(m.iter().map(|(k, v)| (k.clone(), Some(v.clone()))).collect()),
            Some(Value::Array(a)) if !a.is_empty() => Criteria::Choice(
                a.iter().map(|x| (x.as_str().map(String::from).unwrap_or_else(|| py_json(x)), None)).collect(),
            ),
            Some(Value::Object(_)) | Some(Value::Array(_)) => return Err(bad("a choice question needs at least one criterion".into())),
            _ => return Err(bad("a choice question takes 'criteria' as a dict of label -> description, or a list of labels".into())),
        },
        QType::Score => match crit {
            Some(Value::Array(a)) if !a.is_empty() => Criteria::Score(a.clone()),
            Some(Value::Array(_)) => return Err(bad("a score question needs at least one level".into())),
            _ => return Err(bad("a score question takes 'criteria' as a list of level descriptions, index 0 first".into())),
        },
        QType::Noul => {
            let (mut f, mut tr) = (None, None);
            match crit {
                None | Some(Value::Null) => {}
                Some(Value::Object(m)) => {
                    for (k, v) in m {
                        match k.to_lowercase().as_str() {
                            "false" => f = Some(v.clone()),
                            "true" => tr = Some(v.clone()),
                            _ => {}
                        }
                    }
                }
                _ => return Err(bad("a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it".into())),
            }
            let labels = match o.get("labels") {
                None => ("false".to_string(), "true".to_string()),
                Some(Value::Object(m)) => {
                    let get = |k: &str| m.get(k).and_then(Value::as_str).map(|s| s.trim().to_string());
                    match (get("false"), get("true"), m.len()) {
                        (Some(a), Some(b), 2) if !a.is_empty() && !b.is_empty() && a != b => (a, b),
                        _ => return Err(bad("noul labels must map exactly 'false' and 'true' to distinct non-empty strings".into())),
                    }
                }
                _ => return Err(bad("noul labels must map exactly 'false' and 'true' to distinct non-empty strings".into())),
            };
            Criteria::Noul { false_: f, true_: tr, labels }
        }
    };
    Ok(Question { id: id.to_string(), t, ins, crit })
}

pub fn parse_questions(questions: &Value) -> Result<Vec<Question>> {
    let m = questions.as_object().ok_or_else(|| BodiError::Invalid("questions must be an object of id -> definition".into()))?;
    m.iter().map(|(id, d)| parse_question(id, d)).collect()
}

pub trait Encoder {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn cls(&self) -> u32;
    fn sep(&self) -> u32;
    fn mask(&self) -> u32;
    fn pad(&self) -> u32;
    fn mask_token(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: QType,
    /// markers[j] holds option `order[j]`.
    pub order: Vec<usize>,
}

/// laya/common.py `build_sequence`:
/// `[CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`.
pub fn build_sequence(
    tok: &dyn Encoder,
    state: &Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    order: Option<&[usize]>,
    truncate_left: bool,
) -> Encoded {
    let mask_tok = tok.mask_token();
    let opts = render_options(q);
    let order: Vec<usize> = order.map(|o| o.to_vec()).unwrap_or_else(|| (0..opts.len()).collect());
    let ins = q.ins.replace(mask_tok, " ");
    let mut head_ids = tok.encode(&format!("{} question: {}", q.t.name(), ins));
    let mut opt_ids: Vec<Vec<u32>> = order
        .iter()
        .map(|&i| {
            let mut v = vec![tok.mask()];
            let mut e = tok.encode(&format!(" {}", opts[i].replace(mask_tok, " ")));
            e.truncate(48);
            v.extend(e);
            v
        })
        .collect();
    let total = |o: &Vec<Vec<u32>>| o.iter().map(Vec::len).sum::<usize>() as i64;
    let mut opt_budget = head_max_len as i64 - total(&opt_ids);
    if opt_budget < 16 {
        let per = 4i64.max((head_max_len as i64 - 16).div_euclid(opt_ids.len().max(1) as i64)) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as i64 - total(&opt_ids);
    }
    head_ids.truncate(opt_budget.max(8) as usize);
    let mut ids = Vec::with_capacity(max_len);
    ids.push(tok.cls());
    ids.extend(head_ids);
    ids.push(tok.sep());
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(tok.sep());
    let room = max_len.saturating_sub(ids.len() + 1);
    let st = tok.encode(&serialize_state(state).replace(mask_tok, " "));
    let st: &[u32] = if truncate_left { &st[st.len().saturating_sub(room)..] } else { &st[..st.len().min(room)] };
    ids.extend_from_slice(st);
    ids.push(tok.sep());
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    Encoded { ids, markers, qtype: q.t, order }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_formatting() {
        assert_eq!(py_json(&json!({"a": 1, "b": [1, "x"], "c": {"d": null}})), r#"{"a": 1, "b": [1, "x"], "c": {"d": null}}"#);
        assert_eq!(py_json(&json!({"b": 1, "a": 2})), r#"{"b": 1, "a": 2}"#, "key order preserved");
        assert_eq!(py_json(&json!("é")), "\"é\"");
        let crit = json!({"world": 1, "sports": 2, "business": 3, "sci_tech": 4});
        let keys: Vec<&String> = crit.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["world", "sports", "business", "sci_tech"], "caller order, not sorted");
    }

    #[test]
    fn python_float_repr() {
        for (f, want) in [
            (1e-7, "1e-07"), (1e16, "1e+16"), (1.5e16, "1.5e+16"), (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"), (0.00001, "1e-05"), (1.0, "1.0"), (-2.5, "-2.5"), (0.1, "0.1"),
            (123456.789, "123456.789"), (0.0, "0.0"), (2.5e-300, "2.5e-300"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }

    #[test]
    fn renders_like_laya() {
        let q = parse_question("t", &json!({"type": "choice", "instructions": "x", "criteria": {"world": "world news", "sports": null, "b": ""}})).unwrap();
        assert_eq!(render_options(&q), vec!["world: world news", "sports", "b"]);
        let q = parse_question("t", &json!({"type": "score", "instructions": "x", "criteria": ["low", {"desc": "hi"}]})).unwrap();
        assert_eq!(render_options(&q), vec!["level 0: low", "level 1: {\"desc\": \"hi\"}"]);
        let q = parse_question("t", &json!({"type": "noul", "instructions": "x"})).unwrap();
        assert_eq!(render_options(&q), vec!["false: no, the statement does not hold", "true: yes, the statement holds"]);
        let q = parse_question("t", &json!({"type": "noul", "instructions": "x", "criteria": {"True": "spam"}, "labels": {"false": "B", "true": "A"}})).unwrap();
        assert_eq!(render_options(&q), vec!["B: no, the statement does not hold", "A: spam"]);
    }

    #[test]
    fn rejects_malformed_like_laya() {
        assert!(parse_question("q", &json!({"type": "choice", "instructions": "x", "criteria": {}})).is_err());
        assert!(parse_question("q", &json!({"type": "score", "instructions": "x", "criteria": {"a": 1}})).is_err());
        assert!(parse_question("q", &json!({"type": "bool", "instructions": "x"})).is_err());
        assert!(parse_question("q", &json!({"type": "noul"})).is_err());
        assert!(parse_question("q", &json!({"type": "choice", "instructions": "x", "criteria": ["a"], "labels": {}})).is_err());
    }
}
