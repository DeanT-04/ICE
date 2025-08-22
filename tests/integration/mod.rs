//! Integration tests for the ultra-fast AI model
//!
//! This module contains comprehensive integration tests that validate
//! the entire system functionality, including:
//! - Core neural network components
//! - Agentic task decomposition and ensemble voting
//! - Model fusion and coordination
//! - Performance and energy constraints
//! - Benchmark validation against HumanEval and GSM8K
//! - Training time validation on RTX 2070 Ti hardware

pub mod agentic_integration_tests;
pub mod training_time_validation;
pub mod training_time_validation_tests;

// Re-export benchmark tests from tests/benchmarks
pub use crate::tests::benchmarks;