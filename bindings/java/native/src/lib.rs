//! JNI entry points for `ai.mahabodi.Bodi`. The engine lives behind a raw pointer held as a Java
//! `long`; Java guarantees single release through AutoCloseable. Errors (and caught panics)
//! become `ai.mahabodi.BodiException`, never an unwind into the JVM.

use std::panic::{catch_unwind, AssertUnwindSafe};

use jni::objects::{JClass, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;

fn throw(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("ai/mahabodi/BodiException", msg);
}

fn get(env: &mut JNIEnv, s: &JString) -> Result<String, String> {
    if s.is_null() {
        return Ok(String::new());
    }
    env.get_string(s).map(|s| s.into()).map_err(|e| e.to_string())
}

#[no_mangle]
pub extern "system" fn Java_ai_mahabodi_Bodi_nativeNew(mut env: JNIEnv, _c: JClass, config: JString) -> jlong {
    let r = get(&mut env, &config).and_then(|cfg| {
        catch_unwind(AssertUnwindSafe(|| mahabodi_core::Bodi::from_json(&cfg).map_err(|e| e.to_string()))).unwrap_or_else(|_| Err("internal panic".into()))
    });
    match r {
        Ok(b) => Box::into_raw(Box::new(b)) as jlong,
        Err(e) => {
            throw(&mut env, &e);
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_mahabodi_Bodi_nativeCall(mut env: JNIEnv, _c: JClass, handle: jlong, method: JString, args: JString) -> jstring {
    let r = (|| -> Result<String, String> {
        if handle == 0 {
            return Err("engine is closed".into());
        }
        let b = unsafe { &*(handle as *const mahabodi_core::Bodi) };
        let m = get(&mut env, &method)?;
        let a = get(&mut env, &args)?;
        catch_unwind(AssertUnwindSafe(|| {
            let args: serde_json::Value = if a.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(&a).map_err(|e| format!("args are not JSON: {e}"))? };
            b.call(&m, &args).map(|v| v.to_string()).map_err(|e| e.to_string())
        }))
        .unwrap_or_else(|_| Err("internal panic".into()))
    })();
    match r.and_then(|s| env.new_string(s).map_err(|e| e.to_string())) {
        Ok(s) => s.into_raw(),
        Err(e) => {
            throw(&mut env, &e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_mahabodi_Bodi_nativeFree(_env: JNIEnv, _c: JClass, handle: jlong) {
    if handle != 0 {
        drop(unsafe { Box::from_raw(handle as *mut mahabodi_core::Bodi) });
    }
}
