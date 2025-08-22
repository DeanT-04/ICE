//! Ultra-Fast AI Model Library
//!
//! Hyper-efficient AI model with SNN-SSM-Liquid NN hybrid architecture.
//! Targets <100M parameters, <100ms inference latency, <50W power consumption.
//!
//! # Architecture
//!
//! The model combines three neural network types:
//! - **Spiking Neural Networks (SNNs)**: Event-driven processing for energy efficiency
//! - **State-Space Models (SSMs)**: Mamba-style sequence processing with linear scaling
//! - **Liquid Neural Networks**: Adaptive dynamics for continuous learning
//!
//! # Features
//!
//! - Zero-hallucination validation with ensemble voting
//! - Agentic task decomposition and parallel execution
//! - MCP (Model Context Protocol) integration
//! - CPU-optimized inference with 4-bit quantization
//! - Genetic algorithm optimization
//!
//! # Example
//!
//! ```rust,no_run
//! use ultra_fast_ai::model::{UltraFastModel, ModelConfig, InferenceConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ModelConfig::default();
//!     let model = UltraFastModel::load(config)?;
//!     
//!     let result = model.infer(
//!         \"Generate a Rust function to calculate fibonacci\",
//!         InferenceConfig::default()
//!     ).await?;
//!     
//!     println!(\"Generated code: {}\", result);
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod training;
pub mod utils;

// Re-export commonly used types
pub use model::{
    UltraFastModel, ModelConfig, InferenceConfig,
    core::HybridLayer,
    agentic::AgentPool,
};

pub use training::{
    TrainingConfig, Trainer,
    datasets::DatasetLoader,
};

pub use utils::{
    perf::PerformanceMonitor,
    config::load_config,
    schemas::ValidationError,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Model parameter count constraint
pub const MAX_PARAMETERS: usize = 100_000_000; // 100M

/// Performance constraints
pub const MAX_INFERENCE_LATENCY_MS: u64 = 100;
pub const MAX_POWER_CONSUMPTION_W: f32 = 50.0;
pub const MAX_TRAINING_TIME_HOURS: u64 = 24;
pub const MAX_VRAM_USAGE_GB: f32 = 8.0;

/// Error types
#[derive(thiserror::Error, Debug)]
pub enum UltraFastAiError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Training failed: {0}")]
    TrainingError(String),
    
    #[error("Performance constraint violation: {0}")]
    PerformanceError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("MCP integration error: {0}")]
    McpError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Network error: {0}")]
    #[cfg(feature = "mcp-integration")]
    NetworkError(#[from] reqwest::Error),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, UltraFastAiError>;

/// Global configuration
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct GlobalConfig {
    /// Model configuration
    pub model: model::ModelConfig,
    
    /// Training configuration
    pub training: training::TrainingConfig,
    
    /// Performance monitoring configuration
    pub performance: utils::perf::PerformanceConfig,
    
    /// MCP server configurations
    pub mcp: model::mcp::McpConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<std::path::PathBuf>,
    pub structured: bool,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            model: model::ModelConfig::default(),
            training: training::TrainingConfig::default(),
            performance: utils::perf::PerformanceConfig::default(),
            mcp: model::mcp::McpConfig::default(),
            logging: LoggingConfig {
                level: "info".to_string(),
                file: None,
                structured: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[test]
    fn test_constraints() {
        assert_eq!(MAX_PARAMETERS, 100_000_000);
        assert_eq!(MAX_INFERENCE_LATENCY_MS, 100);
        assert_eq!(MAX_POWER_CONSUMPTION_W, 50.0);
        assert_eq!(MAX_TRAINING_TIME_HOURS, 24);
        assert_eq!(MAX_VRAM_USAGE_GB, 8.0);
    }
    
    #[test]
    fn test_default_config() {
        let config = GlobalConfig::default();
        assert_eq!(config.logging.level, "info");
        assert!(!config.logging.structured);
    }
    
    #[test]
    fn test_error_types() {
        let error = UltraFastAiError::ModelLoadError("test".to_string());
        assert!(error.to_string().contains("Model loading failed"));
    }
}