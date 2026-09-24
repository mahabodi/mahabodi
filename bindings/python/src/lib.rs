//! Native module `mahabodi._mahabodi`. The Python package wraps it with typed methods; everything
//! crosses the boundary as JSON text, and the GIL is released while Bodi works.

use mahabodi_core::Bodi;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(frozen)]
struct Engine(Bodi);

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (config_json=""))]
    fn new(py: Python<'_>, config_json: &str) -> PyResult<Self> {
        let cfg = config_json.to_string();
        py.detach(move || Bodi::from_json(&cfg)).map(Engine).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Call a Bodi method with JSON args; returns JSON text. Raises ValueError on failure.
    fn call_json(&self, py: Python<'_>, method: &str, args_json: &str) -> PyResult<String> {
        let (m, a) = (method.to_string(), args_json.to_string());
        py.detach(|| {
            let args: serde_json::Value = if a.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(&a).map_err(|e| e.to_string())? };
            self.0.call(&m, &args).map(|v| v.to_string()).map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }
}

#[pymodule]
fn _mahabodi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
