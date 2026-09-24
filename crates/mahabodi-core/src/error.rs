use thiserror::Error;

#[derive(Debug, Error)]
pub enum BodiError {
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("model not loaded: call load_laya first")]
    NoModel,
    #[error("model error: {0}")]
    Model(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, BodiError>;
