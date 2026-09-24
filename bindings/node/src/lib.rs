//! Node.js binding (napi-rs). `callJson` runs on the calling thread; `callJsonAsync` runs on
//! the libuv thread pool and returns a Promise, so model calls never block the event loop.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

fn run(b: &mahabodi_core::Bodi, method: &str, args: &str) -> std::result::Result<String, String> {
    let args: serde_json::Value = if args.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(args).map_err(|e| e.to_string())? };
    b.call(method, &args).map(|v| v.to_string()).map_err(|e| e.to_string())
}

#[napi(js_name = "Engine")]
pub struct Engine {
    inner: Arc<mahabodi_core::Bodi>,
}

pub struct CallTask {
    inner: Arc<mahabodi_core::Bodi>,
    method: String,
    args: String,
}

impl Task for CallTask {
    type Output = String;
    type JsValue = String;
    fn compute(&mut self) -> Result<String> {
        run(&self.inner, &self.method, &self.args).map_err(|e| Error::new(Status::GenericFailure, e))
    }
    fn resolve(&mut self, _env: Env, out: String) -> Result<String> {
        Ok(out)
    }
}

#[napi]
impl Engine {
    #[napi(constructor)]
    pub fn new(config_json: Option<String>) -> Result<Self> {
        mahabodi_core::Bodi::from_json(config_json.as_deref().unwrap_or(""))
            .map(|b| Engine { inner: Arc::new(b) })
            .map_err(|e| Error::new(Status::InvalidArg, e.to_string()))
    }

    #[napi]
    pub fn call_json(&self, method: String, args_json: String) -> Result<String> {
        run(&self.inner, &method, &args_json).map_err(|e| Error::new(Status::GenericFailure, e))
    }

    #[napi]
    pub fn call_json_async(&self, method: String, args_json: String) -> AsyncTask<CallTask> {
        AsyncTask::new(CallTask { inner: self.inner.clone(), method, args: args_json })
    }
}

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
