//! Non-Latin script guard: a SUBSET of Laya's Router (laya/lang.py `analyse`).
//!
//! Laya's English checkpoint cannot read non-Latin scripts and stays confident while
//! failing (Khmer: 0.000 accuracy at 0.952 confidence, per Laya's own benchmarks), so
//! confidence gating cannot catch it. Bodi flags such inputs *before* inference and
//! forces a handoff instead of returning a confident answer.
//!
//! Scope, stated plainly: only Laya's non-Latin-fraction test (threshold 0.2) is ported.
//! NOT ported: the Latin-script is-English heuristic (stopwords/diacritics), so German,
//! Spanish, French etc. are NOT detected and do reach the English checkpoint (Laya reports
//! it degrades there, e.g. MASSIVE French 0.487 vs English 0.783, rather than collapsing);
//! nor the mixed-text rule (NON_LATIN_MIN_FRACTION with proper-name/IPA exclusions).

use serde::Serialize;

/// Laya's threshold (laya/lang.py `NON_LATIN_FRACTION`).
pub const NON_LATIN_FRACTION: f64 = 0.2;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScriptProfile {
    pub letters: usize,
    pub non_latin_fraction: f64,
    /// Dominant non-Latin script, if any.
    pub script: &'static str,
}

fn script_of(c: char) -> Option<&'static str> {
    let u = c as u32;
    let r = |ranges: &[(u32, u32)]| ranges.iter().any(|&(a, b)| (a..=b).contains(&u));
    Some(match () {
        _ if c.is_ascii_alphabetic()
            || r(&[(0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)]) => "latin",
        _ if r(&[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]) => "greek",
        _ if r(&[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]) => "cyrillic",
        _ if r(&[(0x0530, 0x058F)]) => "armenian",
        _ if r(&[(0x0590, 0x05FF)]) => "hebrew",
        _ if r(&[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]) => "arabic",
        _ if r(&[(0x0900, 0x0DFF), (0xA8E0, 0xA8FF)]) => "indic",
        _ if r(&[(0x0E00, 0x0E7F)]) => "thai",
        _ if r(&[(0x0E80, 0x0EFF)]) => "lao",
        _ if r(&[(0x0F00, 0x0FFF)]) => "tibetan",
        _ if r(&[(0x1000, 0x109F)]) => "myanmar",
        _ if r(&[(0x10A0, 0x10FF)]) => "georgian",
        _ if r(&[(0x1200, 0x137F)]) => "ethiopic",
        _ if r(&[(0x1780, 0x17FF)]) => "khmer",
        _ if r(&[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]) => "hangul",
        _ if r(&[(0x3040, 0x30FF), (0x31F0, 0x31FF)]) => "kana",
        _ if r(&[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]) => "han",
        _ if c.is_alphabetic() => "other",
        _ => return None,
    })
}

pub fn profile(text: &str) -> ScriptProfile {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut letters = 0;
    for c in text.chars() {
        if let Some(s) = script_of(c) {
            letters += 1;
            match counts.iter_mut().find(|(k, _)| *k == s) {
                Some((_, n)) => *n += 1,
                None => counts.push((s, 1)),
            }
        }
    }
    let latin = counts.iter().find(|(k, _)| *k == "latin").map_or(0, |x| x.1);
    let dominant = counts.iter().filter(|(k, _)| *k != "latin").max_by_key(|x| x.1).map(|x| x.0);
    let frac = if letters == 0 { 0.0 } else { 1.0 - latin as f64 / letters as f64 };
    ScriptProfile { letters, non_latin_fraction: (frac * 1e4).round() / 1e4, script: if frac >= NON_LATIN_FRACTION { dominant.unwrap_or("other") } else { "latin" } }
}

/// True when an English-only checkpoint must not be trusted on `text`.
pub fn unreadable_for_english(text: &str) -> bool {
    profile(text).non_latin_fraction >= NON_LATIN_FRACTION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_scripts() {
        assert!(!unreadable_for_english("I was charged twice, please refund"));
        // Latin-script non-English is NOT caught (documented limitation, unlike Laya's Router)
        assert!(!unreadable_for_english("Mein Konto wurde zweimal belastet, bitte erstatten Sie"));
        assert!(!unreadable_for_english("ＲＥＦＵＮＤ ＰＬＥＡＳＥ"), "fullwidth Latin is Latin");
        assert!(unreadable_for_english("ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង"));
        assert_eq!(profile("ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង").script, "khmer");
        assert!(unreadable_for_english("报销需要收据"));
        assert!(unreadable_for_english("Мой счёт списали дважды"));
        // a stray symbol does not flip English text
        assert!(!unreadable_for_english("Set α to 0.05 for the significance test in this experiment"));
    }
}
