//! Schema validation utilities

/// Validation error type
#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("Schema validation failed: {0}")]
    SchemaError(String),
    
    #[error("Type validation failed: {0}")]
    TypeError(String),
    
    #[error("Range validation failed: {0}")]
    RangeError(String),
}

/// Validation result
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;