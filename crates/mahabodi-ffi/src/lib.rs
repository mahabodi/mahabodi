//! C ABI for Bodi. Every call takes and returns UTF-8 JSON, so a binding needs only four
//! symbols. Returned strings are owned by the caller and must be freed with
//! `bodi_string_free`. Panics are caught at the boundary and returned as errors: they never
//! unwind into the host.
//!
//! Result envelope: `{"ok": <value>}` or `{"error": "<message>"}`.

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use mahabodi_core::Bodi;
use serde_json::{json, Value};

pub struct BodiHandle(Bodi);

fn to_c(v: Value) -> *mut c_char {
    CString::new(v.to_string()).unwrap_or_else(|_| CString::new("{\"error\":\"nul byte in output\"}").unwrap()).into_raw()
}

unsafe fn read(s: *const c_char) -> Result<String, String> {
    if s.is_null() {
        return Ok(String::new());
    }
    CStr::from_ptr(s).to_str().map(String::from).map_err(|e| format!("input is not UTF-8: {e}"))
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "panic".into())
}

/// Create an engine from a JSON config (may be NULL or "" for defaults). On failure returns
/// NULL and, if `err` is non-NULL, stores an error string there (free with `bodi_string_free`).
#[no_mangle]
pub unsafe extern "C" fn bodi_new(config_json: *const c_char, err: *mut *mut c_char) -> *mut BodiHandle {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<Bodi, String> {
        let cfg = read(config_json)?;
        Bodi::from_json(&cfg).map_err(|e| e.to_string())
    }));
    let msg = match r {
        Ok(Ok(b)) => return Box::into_raw(Box::new(BodiHandle(b))),
        Ok(Err(e)) => e,
        Err(p) => panic_msg(p),
    };
    if !err.is_null() {
        *err = CString::new(msg).unwrap_or_default().into_raw();
    }
    std::ptr::null_mut()
}

/// Call `method` with JSON `args`. Always returns a JSON envelope string (never NULL).
/// Safe to call concurrently on the same handle from multiple threads.
#[no_mangle]
pub unsafe extern "C" fn bodi_call(h: *const BodiHandle, method: *const c_char, args_json: *const c_char) -> *mut c_char {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<Value, String> {
        let h = h.as_ref().ok_or("null handle")?;
        let method = read(method)?;
        let args = read(args_json)?;
        let args: Value = if args.trim().is_empty() { json!({}) } else { serde_json::from_str(&args).map_err(|e| format!("args are not JSON: {e}"))? };
        h.0.call(&method, &args).map_err(|e| e.to_string())
    }));
    to_c(match r {
        Ok(Ok(v)) => json!({"ok": v}),
        Ok(Err(e)) => json!({"error": e}),
        Err(p) => json!({"error": format!("internal panic: {}", panic_msg(p))}),
    })
}

#[no_mangle]
pub unsafe extern "C" fn bodi_free(h: *mut BodiHandle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[no_mangle]
pub unsafe extern "C" fn bodi_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Library version (static string, do not free).
#[no_mangle]
pub extern "C" fn bodi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_c_abi() {
        unsafe {
            let h = bodi_new(std::ptr::null(), std::ptr::null_mut());
            assert!(!h.is_null());
            let m = CString::new("ingest").unwrap();
            let a = CString::new(r#"{"text":"Refunds take five days. Escalate after two days.","format":"text"}"#).unwrap();
            let out = bodi_call(h, m.as_ptr(), a.as_ptr());
            let v: Value = serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap();
            bodi_string_free(out);
            assert!(v.get("ok").is_some(), "{v}");
            let m = CString::new("query").unwrap();
            let a = CString::new(r#"{"q":"refunds"}"#).unwrap();
            let out = bodi_call(h, m.as_ptr(), a.as_ptr());
            let v: Value = serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap();
            bodi_string_free(out);
            assert_eq!(v["ok"]["matched"], true);
            let bad = CString::new("{not json").unwrap();
            let out = bodi_call(h, m.as_ptr(), bad.as_ptr());
            let v: Value = serde_json::from_str(CStr::from_ptr(out).to_str().unwrap()).unwrap();
            bodi_string_free(out);
            assert!(v["error"].as_str().unwrap().contains("not JSON"));
            let out = bodi_call(std::ptr::null(), m.as_ptr(), a.as_ptr());
            assert!(CStr::from_ptr(out).to_str().unwrap().contains("null handle"));
            bodi_string_free(out);
            bodi_free(h);
            let mut err: *mut c_char = std::ptr::null_mut();
            let bad_cfg = CString::new("{\"auto_density\": 5}").unwrap();
            assert!(bodi_new(bad_cfg.as_ptr(), &mut err).is_null());
            assert!(!err.is_null());
            bodi_string_free(err);
        }
    }
}
